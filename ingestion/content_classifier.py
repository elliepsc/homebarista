"""
ContentClassifier
=================
Classifies coffee video documents by domain, method, and difficulty.
Used by run_ingestion.py to filter low-confidence (non-coffee) content.
"""

import re
from typing import Optional

DOMAIN_SIGNALS: dict[str, list[str]] = {
    "extraction": [
        "extraction", "over-extracted", "under-extracted",
        "yield", "TDS", "EY", "ratio",
    ],
    "grind": [
        "grind", "burr", "grinder", "coarseness", "particle size",
    ],
    "machine": [
        "machine", "boiler", "pressure", "pump", "portafilter", "basket",
    ],
    "origin": [
        "Ethiopian", "Colombian", "Brazilian", "washed", "natural", "terroir",
    ],
    "method": [
        "espresso", "moka", "v60", "aeropress", "french press", "pour over",
    ],
    "troubleshooting": [
        "bitter", "sour", "acidic", "bland", "weak", "channeling", "crema",
    ],
    "technique": [
        "tamping", "distribution", "bloom", "pre-infusion", "flow rate",
    ],
}

# super_automatic is listed first; priority over espresso enforced in _detect_method
METHOD_SIGNALS: dict[str, list[str]] = {
    "super_automatic": [
        "super automatic", "superautomatic", "DeLonghi", "Jura", "Saeco",
        "Philips", "Melitta", "bean to cup", "fully automatic",
    ],
    "espresso":     ["espresso"],
    "moka":         ["moka"],
    "v60":          ["v60"],
    "aeropress":    ["aeropress"],
    "french_press": ["french press"],
}

BEGINNER_SIGNALS = ["beginner", "start", "basic", "introduction"]
ADVANCED_SIGNALS = [
    "SCA", "TDS", "EY", "refractometer", "barista championship", "Scott Rao",
]

_FIELD_WEIGHTS = {"title": 4, "description": 2, "tags": 3, "transcript": 1}


class ContentClassifier:
    """
    Classifies coffee video documents by domain, method, and difficulty.
    Keyword scoring with weighted fields; LLM fallback when confidence < 0.3.
    """

    def classify(self, doc: dict) -> dict:
        """
        Input:  document dict (title, description, tags,
                transcript_text or content)
        Output: {domain, method, difficulty, classification_confidence}

        classification_confidence in [0.0, 1.0].
        confidence = top_domain_score / sum_all_scores.
        """
        fields = self._extract_fields(doc)
        domain_scores = self._score_signals(DOMAIN_SIGNALS, fields)
        total = sum(domain_scores.values())

        if total == 0:
            confidence = 0.0
            top_domain = "general"
        else:
            top_domain = max(domain_scores, key=lambda d: domain_scores[d])
            confidence = min(1.0, domain_scores[top_domain] / total)

        method = self._detect_method(fields)
        difficulty = self._detect_difficulty(fields)

        if confidence < 0.3:
            llm = self._classify_with_llm(doc)
            if llm:
                top_domain = llm.get("domain", top_domain)
                method = llm.get("method", method)

        return {
            "domain": top_domain,
            "method": method,
            "difficulty": difficulty,
            "classification_confidence": confidence,
        }

    def classify_batch(self, docs: list[dict]) -> list[dict]:
        """Update each doc with domain, method, difficulty, confidence."""
        for doc in docs:
            doc.update(self.classify(doc))
        return docs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_fields(doc: dict) -> dict[str, str]:
        tags = doc.get("tags", [])
        tags_text = (
            " ".join(tags) if isinstance(tags, list) else str(tags or "")
        )
        transcript = (
            doc.get("transcript_text") or doc.get("content", "")
        )[:1000]
        return {
            "title":       doc.get("title", ""),
            "description": doc.get("description", ""),
            "tags":        tags_text,
            "transcript":  transcript,
        }

    @staticmethod
    def _count_signal(signal: str, text: str) -> int:
        if not text:
            return 0
        pattern = r"\b" + re.escape(signal) + r"\b"
        return len(re.findall(pattern, text, re.IGNORECASE))

    def _score_signals(
        self,
        signals_map: dict[str, list[str]],
        fields: dict[str, str],
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for category, signals in signals_map.items():
            score = 0.0
            for signal in signals:
                for field, weight in _FIELD_WEIGHTS.items():
                    score += self._count_signal(signal, fields[field]) * weight
            scores[category] = score
        return scores

    def _detect_method(self, fields: dict[str, str]) -> str:
        scores = self._score_signals(METHOD_SIGNALS, fields)
        if not any(scores.values()):
            return "general"
        # super_automatic wins over espresso when both score
        if (
            scores.get("super_automatic", 0) > 0
            and scores.get("espresso", 0) > 0
        ):
            return "super_automatic"
        return max(scores, key=lambda m: scores[m])

    @staticmethod
    def _detect_difficulty(fields: dict[str, str]) -> str:
        lower = " ".join(fields.values()).lower()

        def word_in(sig: str) -> bool:
            return bool(
                re.search(r"\b" + re.escape(sig.lower()) + r"\b", lower)
            )

        if any(word_in(sig) for sig in ADVANCED_SIGNALS):
            return "advanced"
        if any(word_in(sig) for sig in BEGINNER_SIGNALS):
            return "beginner"
        return "intermediate"

    def _classify_with_llm(self, doc: dict) -> Optional[dict]:
        """LLM fallback for low-confidence docs. Returns {domain, method} or None."""
        try:
            # Provider/model/key from .env — see engine/llm_client.py
            from engine.llm_client import LLMClient, extract_json, get_api_key

            if not get_api_key():
                return None

            client = LLMClient()
            fields = self._extract_fields(doc)

            domain_list = list(DOMAIN_SIGNALS.keys())
            method_list = list(METHOD_SIGNALS.keys()) + ["general"]

            prompt = (
                "Classify this coffee video. Reply with JSON only.\n"
                f"Domains: {domain_list}\nMethods: {method_list}\n\n"
                'Format: {"domain": "...", "method": "..."}\n\n'
                f"Title: {fields['title']}\n"
                f"Description: {fields['description'][:200]}\n"
                f"Transcript excerpt: {fields['transcript'][:300]}"
            )

            response = client.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
            )
            return extract_json(response.text)
        except Exception:
            pass
        return None


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    classifier = ContentClassifier()

    samples = [
        {
            "title": "Why Your Espresso Is Bitter",
            "description": "Fix bitter espresso — grind and extraction.",
            "tags": ["bitter", "espresso", "troubleshooting"],
            "transcript_text": (
                "Bitter espresso is caused by over-extraction. "
                "Channeling also creates bitter spots."
            ),
        },
        {
            "title": "DeLonghi Dinamica Setup Guide",
            "description": "Perfect espresso from your DeLonghi.",
            "tags": ["DeLonghi", "espresso", "super automatic"],
            "transcript_text": (
                "The DeLonghi grinds automatically for perfect espresso."
            ),
        },
        {
            "title": "Best hiking trails in Patagonia",
            "description": "Mountain travel vlog.",
            "tags": ["hiking", "travel"],
            "transcript_text": "Beautiful scenery and great mountain views.",
        },
    ]

    for doc in samples:
        result = classifier.classify(doc)
        print(
            f"{doc['title'][:40]:40s} → "
            f"domain={result['domain']:15s} "
            f"method={result['method']:15s} "
            f"diff={result['difficulty']:12s} "
            f"conf={result['classification_confidence']:.2f}"
        )
