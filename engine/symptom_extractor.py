"""
Symptom Extractor
=================
Parses free-form user descriptions → BrewingContext.

Fixes vs. original plan:
1. SYMPTOM_KEYWORDS expanded from 6 to 12 symptoms.
2. LLM fallback now covers BOTH unknown machine AND unrecognised symptoms
   (not just machine_type). ~30% of real inputs won't match keyword rules.
3. Goal detection covers all 4 goals with explicit flow routing.
4. Regex patterns made robust to common variants (22s / 22 sec / 22 seconds).
"""

import re
from typing import Optional

from engine.llm_client import LLMClient, extract_json
from engine.models import BrewingContext


# ------------------------------------------------------------------
# Keyword maps
# ------------------------------------------------------------------

MACHINE_KEYWORDS: dict[str, list[str]] = {
    "super_automatic": [
        "delonghi", "dinamica", "magnifica", "jura", "philips", "saeco",
        "bean to cup", "bean-to-cup", "super automatic", "super-automatic",
        "fully automatic", "automatique", "automatic grinder",
    ],
    "moka": [
        "moka", "bialetti", "stovetop", "stove top", "moka pot",
        "italienne", "cafetière italienne", "percolator",
    ],
    "v60": [
        "v60", "pour over", "pour-over", "hario", "dripper",
        "filter coffee", "filter brew", "wave", "kalita",
    ],
    "aeropress": ["aeropress", "aero press", "aerobie"],
    "french_press": [
        "french press", "french-press", "cafetière à piston",
        "plunger", "press pot", "bodum",
    ],
    "semi_automatic": [
        "breville", "la marzocco", "rancilio", "gaggia", "lelit",
        "ecm", "rocket", "portafilter", "semi automatic", "semi-automatic",
        "prosumer",
    ],
    "manual_lever": ["lever", "flair", "robot", "picopresso", "manual espresso"],
    "nespresso": [
        "nespresso", "vertuo", "original line", "capsule machine",
        "pod machine", "pod coffee",
    ],
}

SYMPTOM_KEYWORDS: dict[str, list[str]] = {
    # Original 6
    "bitter": ["bitter", "amer", "âcre", "harsh", "burnt taste", "brûlé", "harsh aftertaste"],
    "sour": ["sour", "acide", "acidic", "acidité", "tart", "piquant", "aigre", "vinegary"],
    "weak_bland": [
        "weak", "bland", "fade", "insipid", "insipide", "watery", "eau chaude",
        "no flavour", "tasteless", "no taste", "sans goût",
    ],
    "thin_crema": [
        "thin crema", "crème fine", "no crema", "pas de crème", "crema absente",
        "pale crema", "crema disparaît", "crema disappears",
    ],
    "channeling": [
        "channeling", "channelling", "uneven extraction", "extraction inégale",
        "blonde streak", "pale streak", "gushing",
    ],
    "bitter_and_sour": ["unbalanced", "déséquilibré", "complex problem", "both bitter and sour"],
    # New additions — fix the 30% unrecognised rate
    "too_strong": [
        "too strong", "trop fort", "trop concentré", "too intense",
        "overwhelming", "trop puissant", "overpowering",
    ],
    "astringent": [
        "astringent", "dry", "mouth dry", "drying", "rough", "chalky", "sec en bouche",
    ],
    "flat_no_aroma": [
        "no aroma", "no smell", "no fragrance", "flat", "stale smell",
        "aucune odeur", "sans arôme",
    ],
    "inconsistent": [
        "inconsistent", "varies", "sometimes good sometimes bad",
        "different every time", "pas régulier", "irrégulier",
    ],
    "too_slow_extraction": [
        "too slow", "choked", "blocked", "no flow", "drips",
        "trop lent", "bouché", "pas de débit",
    ],
    "too_fast_extraction": [
        "too fast", "gushing", "watery shot", "runs too fast",
        "trop rapide", "extraction rapide",
    ],
}

ORIGIN_KEYWORDS: list[str] = [
    "ethiopian", "yirgacheffe", "sidama", "harrar",
    "colombian", "colombia", "huila", "nariño",
    "brazilian", "brazil", "cerrado", "santos",
    "kenyan", "kenya", "aa",
    "guatemalan", "guatemala",
    "peruvian", "peru",
    "rwandan", "rwanda",
    "indonesian", "sumatra", "java", "sulawesi",
    "vietnamese", "vietnam",
]

ROAST_KEYWORDS: dict[str, list[str]] = {
    "light": ["light roast", "light-roast", "light roasted", "blonde", "clair", "torréfaction légère"],
    "medium": ["medium roast", "medium-roast", "medium roasted", "medium light", "moyen"],
    "dark": ["dark roast", "dark-roast", "dark roasted", "espresso roast", "foncé", "torréfaction foncée"],
}

OPTIMIZE_SIGNALS = ["improve", "better", "optimise", "optimize", "améliorer", "mieux", "perfect"]
EXPLORE_SIGNALS  = ["try", "new", "explore", "discover", "essayer", "nouveau", "comprendre", "understand"]
LEARN_SIGNALS    = ["learn", "beginner", "start", "apprendre", "débutant", "commencer", "how to make"]

# ------------------------------------------------------------------
# Out-of-scope / general question detection
# These route to goal="general" so the agent uses answer_general_question
# instead of the troubleshoot/optimize tool flow.
# ------------------------------------------------------------------

# Machine error codes: E5, E:04, err01, descale alert, clean-me light, etc.
ERROR_CODE_RE = re.compile(
    r'\b(?:error|err|e)[:\s.]?\d{1,2}\b'
    r'|descale\s*(?:alert|warning|light|needed|required|me|indicator)'
    r'|clean\s*(?:me|light|cycle|alert|needed|required)'
    r'|machine\s*(?:is\s*)?(?:beeping|not\s*working|broken|dead|won\'t\s*start|stopped\s*working)'
    r'|\bE\d\b',
    re.IGNORECASE,
)

# Purchase / equipment recommendation signals
PURCHASE_SIGNALS = [
    "should i buy", "recommend a machine", "which machine", "best machine",
    "worth buying", "thinking of buying", "looking to buy", "upgrade to",
    "considering a", " vs ", " versus ", "compare machines", "what machine",
    "buy a grinder", "which grinder", "recommend a grinder", "espresso machine recommendation",
]

# Pure coffee science / theory questions — LLM answers from training knowledge
SCIENCE_SIGNALS = [
    "what is tds", "what is extraction yield", "what is bloom", "what is bypass",
    "what is channeling", "what is the maillard", "what is pre-infusion",
    "explain the science", "chemistry of coffee", "why does coffee taste",
    "how does extraction work", "what causes over-extraction", "explain extraction",
    "what is a ratio", "what does tds mean", "what is solubles",
    "what is the golden ratio", "what is ristretto", "what is lungo",
    "what is a flat white", "difference between espresso and",
]

# Recipe-only requests (no problem, just want a starting point)
RECIPE_SIGNALS = [
    "give me a recipe", "what recipe", "starting recipe", "recipe for v60",
    "recipe for aeropress", "recipe for french press", "recipe for espresso",
    "default recipe", "best recipe", "recommended recipe", "a good recipe",
]


# ------------------------------------------------------------------
# Regex patterns
# ------------------------------------------------------------------

TIME_RE    = re.compile(r"(\d+(?:\.\d+)?)\s*(?:sec(?:onds?)?|s\b)", re.IGNORECASE)
DOSE_RE    = re.compile(r"(\d+(?:\.\d+)?)\s*(?:g(?:r(?:ams?)?)?)\b", re.IGNORECASE)
TEMP_RE    = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°C|degrees?\s*C|celsius)", re.IGNORECASE)
RATIO_RE   = re.compile(r"1\s*:\s*(\d+(?:\.\d+)?)")
GRIND_RE   = re.compile(
    r"(fine|medium|coarse|grossi(?:er|ère)?|fin(?:e)?|"
    r"setting\s*\d+|notch\s*\d+|cran\s*\d+|\d+\s*(?:cran|notch|setting))",
    re.IGNORECASE,
)
FRESHNESS_RE = re.compile(
    r"(\d+)\s*(?:days?|jours?)\s*(?:ago|since|old|après|depuis)?"
    r"|roasted\s*(\d+)\s*(?:days?|weeks?)\s*ago"
    r"|(\d+)\s*week(?:s)?\s*(?:old|ago)",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Extractor
# ------------------------------------------------------------------

class SymptomExtractor:
    """
    Parses free-form coffee problem descriptions into structured BrewingContext.

    Strategy:
    1. Rule-based extraction for everything we can detect with keywords + regex.
    2. LLM fallback (provider/model from .env, see engine/llm_client.py) when
       machine_type is unknown OR when zero symptoms were detected.
    3. Goal detection from signal words.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        demo_mode: bool = False,
    ):
        self.demo_mode = demo_mode
        # Demo mode never builds an LLM client — zero SDK imports, zero network.
        self.client = llm_client if llm_client is not None else (
            None if demo_mode else LLMClient()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, raw_text: str) -> BrewingContext:
        """
        Parse raw user input → BrewingContext.
        Applies rules first, LLM fallback where needed.
        """
        lower = raw_text.lower()

        machine_type = self._extract_machine_type(lower)
        symptoms = self._extract_symptoms(lower)
        extraction_time = self._extract_time(raw_text)
        dose = self._extract_dose(raw_text)
        temp = self._extract_temp(raw_text)
        grind = self._extract_grind(lower)
        origin = self._extract_origin(lower)
        roast = self._extract_roast(lower)
        freshness = self._extract_freshness(raw_text)
        ratio = self._extract_ratio(raw_text)
        goal = self._detect_goal(lower, symptoms)

        # Specs-based detection: user provides parameters without a symptom.
        # "18g, 93°C, 28 seconds, how do I improve?" → route to optimize.
        has_specs = any([extraction_time, dose, temp, grind, ratio])
        if not symptoms and has_specs and goal not in ("learn", "explore"):
            goal = "optimize"

        # LLM fallback: trigger when machine unknown OR no symptoms AND no optimize route
        needs_llm = (machine_type == "unknown") or (not symptoms and goal == "troubleshoot")
        if needs_llm and not self.demo_mode:
            llm_result = self._llm_fallback(raw_text)
            if machine_type == "unknown":
                machine_type = llm_result.get("machine_type", "unknown")
            if not symptoms and goal == "troubleshoot":
                symptoms = llm_result.get("symptoms", [])
                # If LLM still finds nothing but user has specs → upgrade to optimize
                if not symptoms and has_specs:
                    goal = "optimize"

        return BrewingContext(
            machine_type=machine_type,
            machine_model=self._extract_model_name(raw_text),
            bean_origin=origin,
            roast_level=roast,
            bean_freshness_days=freshness,
            grind_size=grind,
            dose_grams=dose,
            water_temp_celsius=temp,
            extraction_time_seconds=extraction_time,
            water_ratio=ratio,
            raw_problem=raw_text,
            goal=goal,
            symptoms_detected=symptoms,
        )

    # ------------------------------------------------------------------
    # Rule-based extractors
    # ------------------------------------------------------------------

    def _extract_machine_type(self, lower: str) -> str:
        for machine_type, keywords in MACHINE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return machine_type
        return "unknown"

    def _extract_symptoms(self, lower: str) -> list[str]:
        found = []
        for symptom, keywords in SYMPTOM_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                found.append(symptom)
        return found

    def _extract_time(self, text: str) -> Optional[int]:
        m = TIME_RE.search(text)
        return int(float(m.group(1))) if m else None

    def _extract_dose(self, text: str) -> Optional[float]:
        m = DOSE_RE.search(text)
        return float(m.group(1)) if m else None

    def _extract_temp(self, text: str) -> Optional[float]:
        m = TEMP_RE.search(text)
        return float(m.group(1)) if m else None

    def _extract_grind(self, lower: str) -> Optional[str]:
        m = GRIND_RE.search(lower)
        return m.group(0) if m else None

    def _extract_origin(self, lower: str) -> Optional[str]:
        for origin in ORIGIN_KEYWORDS:
            if origin in lower:
                return origin.title()
        return None

    def _extract_roast(self, lower: str) -> Optional[str]:
        for roast_level, keywords in ROAST_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return roast_level
        return None

    def _extract_freshness(self, text: str) -> Optional[int]:
        m = FRESHNESS_RE.search(text)
        if not m:
            return None
        if m.group(1):
            return int(m.group(1))
        if m.group(2):
            return int(m.group(2))
        if m.group(3):
            return int(m.group(3)) * 7
        return None

    def _extract_ratio(self, text: str) -> Optional[str]:
        m = RATIO_RE.search(text)
        return f"1:{m.group(1)}" if m else None

    def _extract_model_name(self, text: str) -> Optional[str]:
        """
        Best-effort extraction of specific model names.
        Looks for known brand+model patterns.
        """
        MODEL_PATTERNS = [
            r"(DeLonghi\s+\w+)",
            r"(Jura\s+[A-Z]\d+)",
            r"(Breville\s+\w+)",
            r"(La Marzocco\s+\w+)",
            r"(Gaggia\s+\w+)",
            r"(Rancilio\s+\w+)",
            r"(Nespresso\s+\w+)",
            r"(Philips\s+\w+)",
        ]
        for pattern in MODEL_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    def _detect_goal(self, lower: str, symptoms: list[str]) -> str:
        # Out-of-scope queries — route to general LLM fallback
        # Check these BEFORE learn/explore so specific intents aren't swallowed
        if ERROR_CODE_RE.search(lower):
            return "general"
        if any(s in lower for s in PURCHASE_SIGNALS):
            return "general"
        if any(s in lower for s in SCIENCE_SIGNALS):
            return "general"
        if any(s in lower for s in RECIPE_SIGNALS) and not symptoms:
            return "general"

        if any(s in lower for s in LEARN_SIGNALS):
            return "learn"
        if any(s in lower for s in EXPLORE_SIGNALS):
            return "explore"
        if any(s in lower for s in OPTIMIZE_SIGNALS):
            return "optimize"
        if symptoms:
            return "troubleshoot"
        return "troubleshoot"  # default

    # ------------------------------------------------------------------
    # LLM fallback
    # ------------------------------------------------------------------

    def _llm_fallback(self, raw_text: str) -> dict:
        """
        Call the configured LLM when rules fail to identify machine or symptoms.
        Returns partial dict to merge with rule-based results.
        """
        prompt = f"""Analyse this coffee problem description and extract structured information.

Description: "{raw_text}"

Respond with ONLY valid JSON (no explanation):
{{
  "machine_type": "<one of: super_automatic|semi_automatic|manual_lever|moka|v60|aeropress|french_press|nespresso|unknown>",
  "symptoms": ["<list of symptoms from: bitter|sour|weak_bland|thin_crema|channeling|too_strong|astringent|flat_no_aroma|inconsistent|too_slow_extraction|too_fast_extraction>"],
  "confidence": <0.0-1.0>
}}"""

        try:
            # 600 not 200: reasoning models on Groq (gpt-oss default, qwen3
            # fallback via LLM_BASE_URL) burn tokens on hidden chain-of-thought
            # before emitting the JSON.
            response = self.client.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
            )
            parsed = extract_json(response.text)
            return parsed if parsed else {"machine_type": "unknown", "symptoms": []}
        except Exception as e:
            print(f"  [LLM fallback failed: {e}]")
            return {"machine_type": "unknown", "symptoms": []}


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    extractor = SymptomExtractor(demo_mode=True)

    cases = [
        "My DeLonghi Dinamica makes bitter espresso, extraction 28 seconds",
        "ma moka donne un café très acide",
        "V60, Ethiopian light roast, tastes too sour, 22s extraction, 15g dose",
        "my coffee is too strong and leaves my mouth dry",  # tests new symptoms
        "I want to learn how to make better espresso",     # tests goal=learn
        "Aeropress, 1:15 ratio, coffee tastes flat, no aroma at all",
    ]

    for case in cases:
        ctx = extractor.extract(case)
        print("Input: " + case[:60] + "...")
        print("  machine=" + ctx.machine_type + " | symptoms=" + str(ctx.symptoms_detected) + " | goal=" + ctx.goal)
        if ctx.extraction_time_seconds:
            print("  time=" + str(ctx.extraction_time_seconds) + "s")
        if ctx.dose_grams:
            print("  dose=" + str(ctx.dose_grams) + "g")
