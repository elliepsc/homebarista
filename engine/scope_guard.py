"""
ScopeGuard — deterministic off-topic filter. ZERO LLM calls, ZERO tokens.
=========================================================================
Every request that is not about coffee / barista / coffee machines is
refused BEFORE any LLM call, any model loading, and any retrieval.

Design rules (see ultraplan v3, Phase A2):
- 100% deterministic: whole-word match against a bilingual (EN+FR)
  coffee vocabulary + machine brand list. Pure Python, no network.
- Multi-turn aware: a short follow-up without a coffee keyword is
  allowed if any PREVIOUS user turn in the conversation was in-scope.
- Length cap: inputs over MAX_INPUT_CHARS are refused deterministically
  (protects against prompt-stuffing and runaway token costs).
- This module must NEVER import anthropic or make network calls —
  its whole reason to exist is to cost 0 tokens.
"""

import re
import unicodedata

REFUSAL_MESSAGE = (
    "☕ HomeBarista only answers questions about coffee: brewing, espresso, "
    "grinders, beans, and coffee machines. For anything else, please use a "
    "general-purpose assistant (ChatGPT, Claude...). Off-topic requests are "
    "not processed and no AI tokens are spent on them."
)

TOO_LONG_MESSAGE = (
    "☕ Your message is too long (max 1500 characters). Please describe your "
    "coffee problem concisely: machine, beans, what tastes wrong."
)

COFFEE_VOCAB = {
    # EN — drinks & methods
    "coffee", "espresso", "latte", "cappuccino", "americano", "ristretto",
    "lungo", "macchiato", "cortado", "flat", "brew", "brewing", "pour",
    "moka", "v60", "chemex", "aeropress", "percolator", "drip", "cold",
    # EN — process & gear
    "grind", "grinder", "burr", "roast", "roaster", "bean", "beans", "crema",
    "tamp", "tamping", "portafilter", "basket", "puck", "dose", "yield",
    "extraction", "channeling", "bloom", "kettle", "barista", "decaf",
    "arabica", "robusta", "filter", "shot", "dial", "dialing",
    # EN — symptoms
    "bitter", "sour", "acidic", "astringent", "watery", "bland", "burnt",
    # FR
    "café", "expresso", "mouture", "moulin", "torréfaction", "grain",
    "grains", "amer", "amère", "acide", "cafetière", "percolateur",
    "crème", "tasse", "infusion", "piston", "capsule", "dosette",
    # brands (machines)
    "delonghi", "breville", "gaggia", "jura", "philips", "sage", "nespresso",
    "rancilio", "lelit", "flair", "bialetti", "krups", "melitta", "smeg",
    "rocket", "profitec", "eureka", "baratza", "niche", "wilfa", "fellow",
}

# Words are lowercase alphabetic sequences, accents included (FR support).
# "v60" is handled by allowing digits inside a word.
_WORD_RE = re.compile(r"[a-zà-ÿ0-9]+")


def _strip_accents(text: str) -> str:
    """Fold diacritics so "cafe" matches "café" — many FR users type without
    accents (mobile keyboards), and refusing "mon cafe est amer" would be a
    false negative on a perfectly on-topic question."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


# Accent-folded vocab, matched against accent-folded input (see _strip_accents).
_COFFEE_VOCAB_FOLDED = {_strip_accents(w) for w in COFFEE_VOCAB}


class ScopeGuard:
    """Deterministic in-scope / out-of-scope classifier. No LLM, no network."""

    MAX_INPUT_CHARS = 1500

    def _has_coffee_word(self, text: str) -> bool:
        folded = _strip_accents(text.lower())
        return any(w in _COFFEE_VOCAB_FOLDED for w in _WORD_RE.findall(folded))

    def check(self, text: str, conversation_history: list[dict] | None = None) -> dict:
        """
        Returns {"in_scope": bool, "reason": "ok" | "off_topic" | "too_long",
                 "message": str}.

        Rules:
        - length cap first: len(text) > MAX_INPUT_CHARS → "too_long"
        - whole-word match on text.lower(): ≥ 1 hit in COFFEE_VOCAB → in-scope
        - 0 hit: scan conversation_history — any prior in-scope user turn
          makes short follow-ups ("yes, 20 seconds") in-scope
        - otherwise → "off_topic"
        """
        if len(text) > self.MAX_INPUT_CHARS:
            return {"in_scope": False, "reason": "too_long", "message": TOO_LONG_MESSAGE}

        if self._has_coffee_word(text):
            return {"in_scope": True, "reason": "ok", "message": ""}

        # No coffee word in this message: allow follow-ups if any PREVIOUS
        # user turn was in-scope (multi-turn conversations).
        for turn in (conversation_history or []):
            if turn.get("role") == "user" and self._has_coffee_word(str(turn.get("content", ""))):
                return {"in_scope": True, "reason": "ok", "message": ""}

        return {"in_scope": False, "reason": "off_topic", "message": REFUSAL_MESSAGE}
