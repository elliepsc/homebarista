"""
Diagnostic Planner — Deterministic Engine
==========================================
Maps symptoms + machine context → root causes + intervention plan.

Fixes vs. original plan:
1. SYMPTOM_MATRIX expanded to 12 symptoms (matches SymptomExtractor).
2. Machine capability filter: interventions physically impossible on a given
   machine are removed before returning (no more "use WDT" for Nespresso).
3. Probability values renamed to heuristic_weight with explicit disclaimer.
4. Multi-symptom handling: when bitter + sour co-occur, diagnose "mixed extraction".
5. Non-troubleshoot goals (learn, explore, optimize) get a dedicated flow,
   not a ValueError.
6. DiagnosticResult now carries method_overrides applied (for transparency).

IMPORTANT NOTE ON HEURISTIC WEIGHTS:
Values like 0.70 are NOT empirically calibrated probabilities.
They are expert heuristic weights encoding relative likelihood
based on extraction science literature (Scott Rao, Barista Hustle, SCA).
They are used for ranking root causes only — not for probabilistic inference.
Present them as such in the README and in interviews.
"""

from dataclasses import field
from typing import Optional

from engine.models import (
    BrewingContext,
    DiagnosticResult,
    Intervention,
    RootCause,
)


# ------------------------------------------------------------------
# Machine capability map
# (which parameters can the user actually adjust?)
# ------------------------------------------------------------------

MACHINE_ADJUSTABLE: dict[str, set[str]] = {
    "super_automatic": {
        "grind_size", "coffee_strength", "temperature_setting", "water_amount",
    },
    "nespresso": {"water_amount", "machine_cleaning"},
    "moka": {"heat_level", "grind_size", "dose", "fill_level"},
    "v60": {"grind_size", "dose", "water_temp", "pour_technique", "bloom_time"},
    "aeropress": {"grind_size", "dose", "water_temp", "steep_time", "pressure"},
    "french_press": {"grind_size", "dose", "water_temp", "steep_time"},
    "semi_automatic": {
        "grind_size", "dose", "water_temp", "pre_infusion",
        "pressure", "tamping", "distribution",
    },
    "manual_lever": {
        "grind_size", "dose", "water_temp", "lever_pressure", "tamping",
    },
    "unknown": {"grind_size", "dose", "water_temp"},
}

# Map Intervention.parameter values → capability keys
PARAM_TO_CAPABILITY: dict[str, str] = {
    "grind_size":    "grind_size",
    "water_temp":    "water_temp",
    "dose":          "dose",
    "time":          "grind_size",   # adjusting time ≈ adjusting grind
    "freshness":     "grind_size",   # always universally applicable
    "maintenance":   "machine_cleaning",
    "tamping":       "tamping",
    "distribution":  "distribution",
    "technique":     "pour_technique",
    "heat_level":    "heat_level",
    "fill_level":    "fill_level",
    "temperature_setting": "temperature_setting",
    "pressure":      "pressure",
    "bloom_time":    "bloom_time",
    "steep_time":    "steep_time",
}

# Parameters always applicable regardless of machine
UNIVERSAL_PARAMS = {"freshness", "machine_cleaning", None}


# ------------------------------------------------------------------
# Symptom matrix (12 symptoms)
# ------------------------------------------------------------------

def _rc(hypothesis, weight, evidence, parameter=None):
    return RootCause(
        hypothesis=hypothesis,
        probability=weight,      # heuristic_weight; named probability for model compat
        evidence=evidence,
        parameter_affected=parameter,
    )

def _iv(step, action, parameter=None, direction=None, magnitude=None,
        expected="", validation="", priority="high"):
    return Intervention(
        step=step, action=action, parameter=parameter,
        direction=direction, magnitude=magnitude,
        expected_result=expected, validation_test=validation, priority=priority,
    )


SYMPTOM_MATRIX: dict[str, dict] = {

    "bitter": {
        "root_causes": [
            _rc("over-extraction",     0.70, "Bitter = too many compounds dissolved; extraction ran too long or grind too fine", "grind_size"),
            _rc("grind_too_fine",      0.65, "Fine grind = high surface area = fast dissolution of bitter compounds", "grind_size"),
            _rc("water_too_hot",       0.50, "High temp extracts chlorogenic acids and other bitter compounds faster", "water_temp"),
            _rc("dark_roast_mismatch", 0.35, "Dark roasts amplify bitterness at standard espresso parameters; need shorter extraction"),
        ],
        "method_overrides": {
            "moka": [
                _rc("heat_too_high",      0.85, "Moka: excessive heat scorches coffee as it passes through grounds", "heat_level"),
                _rc("too_long_on_flame",  0.75, "Leaving moka on flame after extraction extracts bitter late-stage compounds"),
                _rc("grind_too_fine",     0.60, "Moka grind should be medium — finer than filter but coarser than espresso", "grind_size"),
            ],
            "super_automatic": [
                _rc("grind_too_fine",          0.70, "Digital grind setting too fine = over-extraction in pressurised brew group", "grind_size"),
                _rc("temperature_setting_high", 0.55, "Temperature set too high in machine settings", "temperature_setting"),
                _rc("coffee_strength_high",    0.45, "Strength/aroma setting too high = higher dose or finer grind", "coffee_strength"),
            ],
            "nespresso": [
                _rc("capsule_type",    0.80, "Intense/dark capsules inherently taste more bitter; try a lower-intensity capsule"),
                _rc("machine_dirty",   0.55, "Old coffee oils in machine amplify bitterness; descale and clean", "maintenance"),
            ],
        },
        "interventions": [
            _iv(1, "Go 1 notch coarser on grind", "grind_size", "coarser", "1 notch",
                "Less surface contact = fewer bitter compounds extracted",
                "Shot should taste less harsh; check if body improves"),
            _iv(2, "Lower water temperature by 2°C", "water_temp", "lower", "2°C",
                "Cooler water extracts fewer bitter compounds",
                "Taste should feel cleaner and less aggressive"),
            _iv(3, "Shorten extraction by 3-5 seconds", "time", "shorter", "3-5s",
                "Less contact time = less bitterness",
                "Extraction should feel less heavy"),
            _iv(4, "Check bean freshness (ideal < 3 weeks post-roast)", "freshness",
                priority="critical"),
        ],
    },

    "sour": {
        "root_causes": [
            _rc("under-extraction",     0.75, "Sour = extraction stopped before acids were balanced by sugars and bitters", "grind_size"),
            _rc("grind_too_coarse",     0.70, "Coarse grind = low surface area = incomplete extraction", "grind_size"),
            _rc("water_too_cold",       0.55, "Low temp doesn't dissolve sugars fully; acids dominate", "water_temp"),
            _rc("too_short_extraction", 0.60, "Extraction time too short to develop sweetness", "time"),
            _rc("light_roast_challenge",0.40, "Light roasts retain more organic acids; require hotter water and longer extraction"),
        ],
        "interventions": [
            _iv(1, "Go 1 notch finer on grind", "grind_size", "finer", "1 notch",
                "More surface area = fuller extraction = more sweetness to balance acids",
                "Acidity should feel brighter but less sharp"),
            _iv(2, "Raise water temperature by 2°C", "water_temp", "higher", "2°C",
                "Higher temp dissolves sugars and balances acids",
                "Coffee should taste rounder and less sharp"),
            _iv(3, "Extend extraction by 3-5 seconds", "time", "longer", "3-5s",
                "More extraction time develops sweetness",
                "Sourness should mellow; look for more body"),
        ],
    },

    "weak_bland": {
        "root_causes": [
            _rc("dose_too_low",    0.70, "Not enough coffee for the water volume = diluted cup", "dose"),
            _rc("grind_too_coarse",0.65, "Coarse grind = fast channeling through grounds = thin extraction", "grind_size"),
            _rc("stale_beans",     0.55, "Old beans have lost CO2 and volatile aromatics = flat taste", "freshness"),
            _rc("under-extraction",0.50, "Extraction didn't capture full flavour range", "grind_size"),
        ],
        "interventions": [
            _iv(1, "Increase dose by 1-2g", "dose", "higher", "1-2g",
                "More coffee material = stronger, more complex extraction",
                "Taste should feel fuller and more present"),
            _iv(2, "Go slightly finer on grind", "grind_size", "finer", "0.5 notch",
                "Slower flow = more contact time = fuller extraction",
                "Coffee should taste less watery"),
            _iv(3, "Check bean freshness — aim for < 3 weeks post-roast", "freshness",
                priority="critical",
                validation="Fresh beans will smell intensely aromatic when ground"),
        ],
    },

    "thin_crema": {
        "root_causes": [
            _rc("stale_beans",     0.75, "Crema = CO2 gas. Old beans have off-gassed = no crema", "freshness"),
            _rc("grind_too_coarse",0.60, "Coarse grind = lower extraction pressure = less emulsification", "grind_size"),
            _rc("low_pressure",    0.50, "Machine pressure below 7 bar can't emulsify oils", "pressure"),
            _rc("machine_dirty",   0.45, "Scale or coffee oil buildup affects flow and emulsification", "maintenance"),
        ],
        "interventions": [
            _iv(1, "Check roast date — should be within 2-3 weeks", "freshness",
                priority="critical",
                validation="Fresh beans produce thick, persistent crema immediately"),
            _iv(2, "Go 1 notch finer on grind", "grind_size", "finer", "1 notch",
                "Finer grind increases resistance = better emulsification",
                "Crema should appear thicker and darker"),
            _iv(3, "Descale and clean group head", "maintenance",
                priority="high",
                validation="Flow should be more consistent after cleaning"),
        ],
    },

    "channeling": {
        "root_causes": [
            _rc("uneven_distribution",  0.75, "Uneven coffee bed = water finds path of least resistance", "distribution"),
            _rc("tamping_uneven",       0.65, "Uneven tamp creates low-density pockets for channeling", "tamping"),
            _rc("dose_inconsistent",    0.50, "Variable dose = inconsistent bed depth and density", "dose"),
            _rc("grind_too_fine_clumps",0.40, "Very fine grinds clump and create uneven distribution", "grind_size"),
        ],
        "interventions": [
            _iv(1, "Use WDT (Weiss Distribution Technique) to break up clumps", "distribution",
                validation="Bed should look flat and even before tamping"),
            _iv(2, "Tamp level with consistent 15-20kg pressure", "tamping",
                validation="No tilted puck after extraction; even wet surface"),
            _iv(3, "Weigh dose to ±0.1g consistency", "dose",
                validation="Same dose every shot removes one variable"),
        ],
    },

    "bitter_and_sour": {
        "root_causes": [
            _rc("mixed_extraction",  0.70, "Some channels over-extract (bitter) while others under-extract (sour) simultaneously", "distribution"),
            _rc("channeling",        0.65, "Uneven water flow creates localised over and under extraction zones", "distribution"),
            _rc("grind_inconsistent",0.55, "Inconsistent grind particle size creates zones of different extraction rates", "grind_size"),
        ],
        "interventions": [
            _iv(1, "Improve distribution before tamping — use WDT tool", "distribution",
                validation="Complex bitter-sour combination should simplify to one primary note"),
            _iv(2, "Check grinder burrs — inconsistent grind suggests worn burrs", "grind_size",
                priority="high"),
            _iv(3, "Adjust grind based on which flavour dominates after fix #1", "grind_size",
                expected="Once distribution is fixed, dominant note guides next step"),
        ],
    },

    "too_strong": {
        "root_causes": [
            _rc("dose_too_high",    0.70, "Too much coffee per unit water = high TDS = overwhelming strength", "dose"),
            _rc("ratio_too_low",    0.65, "Not enough water for the dose; yield too low", "dose"),
            _rc("grind_too_fine",   0.50, "Fine grind = slow extraction = higher TDS at same volume", "grind_size"),
        ],
        "interventions": [
            _iv(1, "Reduce dose by 1-2g", "dose", "lower", "1-2g",
                "Less coffee = lower strength",
                "Taste should feel less intense but preserve flavour complexity"),
            _iv(2, "Increase water output / yield", "dose", "higher", "10-15ml more",
                "More dilution = lower TDS",
                "Coffee should feel lighter and more drinkable"),
        ],
    },

    "astringent": {
        "root_causes": [
            _rc("over-extraction",  0.70, "Astringency = polyphenols and tannins extracted too aggressively", "grind_size"),
            _rc("water_too_hot",    0.60, "High temp extracts astringent compounds faster", "water_temp"),
            _rc("dark_roast",       0.45, "Dark roasts are more prone to astringency when over-extracted"),
        ],
        "interventions": [
            _iv(1, "Go 1 notch coarser on grind", "grind_size", "coarser", "1 notch",
                "Reduces astringent compound extraction",
                "Dry/rough mouthfeel should improve noticeably"),
            _iv(2, "Lower water temperature by 2-3°C", "water_temp", "lower", "2-3°C",
                "Cooler water is gentler on extraction",
                "Taste should feel cleaner and smoother"),
        ],
    },

    "flat_no_aroma": {
        "root_causes": [
            _rc("stale_beans",      0.85, "Aromatics are volatile — old beans have lost them through off-gassing", "freshness"),
            _rc("grind_too_coarse", 0.50, "Coarse grind releases less aromatic surface area", "grind_size"),
            _rc("water_too_cold",   0.40, "Cool water doesn't volatilise aromatic compounds into the cup", "water_temp"),
        ],
        "interventions": [
            _iv(1, "Replace beans — freshness is the primary lever here", "freshness",
                priority="critical",
                validation="Freshly ground coffee should smell intensely floral, fruity, or nutty"),
            _iv(2, "Grind immediately before brewing — never pre-grind", "grind_size",
                priority="high",
                validation="Aroma should be noticeably stronger when grinding fresh"),
        ],
    },

    "inconsistent": {
        "root_causes": [
            _rc("dose_variable",    0.70, "Without a scale, dose varies shot to shot", "dose"),
            _rc("grind_drift",      0.60, "Grinder burrs or settings drifting between sessions", "grind_size"),
            _rc("technique_variable",0.50, "Tamping pressure or distribution technique inconsistent", "tamping"),
        ],
        "interventions": [
            _iv(1, "Use a scale — weigh dose to ±0.1g every shot", "dose",
                priority="critical",
                validation="Consistency will visibly improve within 5 shots"),
            _iv(2, "Record grind setting and don't change it between sessions", "grind_size",
                validation="Shots should become predictable within a session"),
            _iv(3, "Standardise tamping technique — same pressure, same motion", "tamping",
                validation="Puck surface should look identical after every tamp"),
        ],
    },

    "too_slow_extraction": {
        "root_causes": [
            _rc("grind_too_fine",    0.80, "Too fine = too much resistance = water can't flow through", "grind_size"),
            _rc("dose_too_high",     0.60, "Too much coffee in basket = restricted flow", "dose"),
            _rc("machine_scale",     0.50, "Scale buildup reduces pump efficiency", "maintenance"),
        ],
        "interventions": [
            _iv(1, "Go 1-2 notches coarser on grind immediately", "grind_size", "coarser", "1-2 notches",
                "Reduces resistance = faster, more even flow",
                "Shot should flow within 5-8 seconds of pump start"),
            _iv(2, "Reduce dose by 1g", "dose", "lower", "1g",
                "Less coffee = less resistance",
                "Flow rate should increase noticeably"),
            _iv(3, "Run a descale cycle if not done in last 3 months", "maintenance",
                priority="high"),
        ],
    },

    "too_fast_extraction": {
        "root_causes": [
            _rc("grind_too_coarse",  0.80, "Too coarse = low resistance = water rushes through without proper extraction", "grind_size"),
            _rc("dose_too_low",      0.55, "Less coffee = less resistance = faster flow", "dose"),
            _rc("tamping_too_light", 0.50, "Insufficient tamping pressure = loose puck = channeling and fast flow", "tamping"),
        ],
        "interventions": [
            _iv(1, "Go 1-2 notches finer on grind", "grind_size", "finer", "1-2 notches",
                "More resistance = slower, more thorough extraction",
                "Shot time should increase; taste should gain body and sweetness"),
            _iv(2, "Tamp with firm, even pressure (~15-20kg)", "tamping",
                validation="Puck should feel solid; no movement when you press it"),
        ],
    },
}


# ------------------------------------------------------------------
# DiagnosticPlanner
# ------------------------------------------------------------------

class DiagnosticPlanner:
    """
    Deterministic diagnostic engine.
    Never calls LLM. Maps symptoms + context → root causes + filtered interventions.
    """

    def diagnose(self, context: BrewingContext) -> DiagnosticResult:
        symptoms = getattr(context, "symptoms_detected", []) or []

        # --- Goal-based routing (no symptoms required) ---

        # general → out-of-scope question (error codes, purchase, science, recipe)
        # Return a minimal non-blocking result; the agent will call answer_general_question
        if context.goal == "general":
            return self._general_result(context)

        # learn / explore → foundational principles for this machine
        if context.goal in ("learn", "explore") and not symptoms:
            return self._learn_explore_result(context)

        # optimize → user has specs, wants to refine (no problem per se)
        if context.goal == "optimize" and not symptoms:
            return self._optimize_result(context)

        # troubleshoot with no symptoms at all
        if not symptoms:
            return self._no_symptoms_result(context)

        # Handle co-occurring bitter + sour as a special case
        if "bitter" in symptoms and "sour" in symptoms:
            symptoms = [s for s in symptoms if s not in ("bitter", "sour")]
            symptoms.insert(0, "bitter_and_sour")

        # Gather root causes from all detected symptoms
        all_root_causes: list[RootCause] = []
        all_interventions: list[Intervention] = []

        for symptom in symptoms:
            matrix_entry = SYMPTOM_MATRIX.get(symptom)
            if not matrix_entry:
                continue

            # Apply method overrides if available
            overrides = matrix_entry.get("method_overrides", {})
            method_key = context.machine_type
            if method_key in overrides:
                causes = overrides[method_key]
            else:
                causes = matrix_entry["root_causes"]

            all_root_causes.extend(causes)
            all_interventions.extend(matrix_entry["interventions"])

        # Deduplicate root causes by hypothesis, keeping highest weight
        seen: dict[str, RootCause] = {}
        for rc in all_root_causes:
            if rc.hypothesis not in seen or rc.probability > seen[rc.hypothesis].probability:
                seen[rc.hypothesis] = rc
        deduped_causes = sorted(seen.values(), key=lambda x: x.probability, reverse=True)

        # Context-based probability boosts
        deduped_causes = self._apply_context_boosts(deduped_causes, context)

        # Take top 3 causes
        top_causes = deduped_causes[:3]

        # Build intervention plan — deduplicate by parameter, max 4 steps
        filtered_interventions = self._filter_interventions(
            all_interventions, context.machine_type
        )

        # Re-number steps
        for i, iv in enumerate(filtered_interventions, start=1):
            iv.step = i

        # Confidence score
        confidence = self._compute_confidence(top_causes, context)

        # Warnings
        warnings = self._build_warnings(context, symptoms)

        return DiagnosticResult(
            symptoms=symptoms,
            root_causes=top_causes,
            intervention_plan=filtered_interventions,
            diagnostic_confidence=confidence,
            method_detected=context.machine_type,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Context boosts
    # ------------------------------------------------------------------

    def _apply_context_boosts(
        self, causes: list[RootCause], context: BrewingContext
    ) -> list[RootCause]:
        """Adjust heuristic weights based on measured parameters."""
        for rc in causes:
            t = context.extraction_time_seconds

            if t and t < 20 and rc.hypothesis == "under-extraction":
                rc.probability = min(0.95, rc.probability + 0.20)
            if t and t > 32 and rc.hypothesis == "over-extraction":
                rc.probability = min(0.95, rc.probability + 0.20)
            if context.bean_freshness_days and context.bean_freshness_days > 30:
                if "stale" in rc.hypothesis:
                    rc.probability = min(0.95, rc.probability + 0.20)
            if context.machine_type == "moka" and rc.hypothesis == "heat_too_high":
                rc.probability = min(0.95, rc.probability + 0.15)

        return sorted(causes, key=lambda x: x.probability, reverse=True)

    # ------------------------------------------------------------------
    # Machine capability filter
    # ------------------------------------------------------------------

    def _filter_interventions(
        self, interventions: list[Intervention], machine_type: str
    ) -> list[Intervention]:
        """
        Remove interventions that are physically impossible on this machine.
        Also deduplicate by parameter, keeping highest-priority item.
        """
        adjustable = MACHINE_ADJUSTABLE.get(machine_type, MACHINE_ADJUSTABLE["unknown"])
        filtered = []
        seen_params: set[str] = set()

        # Sort critical first
        sorted_ivs = sorted(
            interventions,
            key=lambda x: (0 if x.priority == "critical" else 1 if x.priority == "high" else 2),
        )

        for iv in sorted_ivs:
            param = iv.parameter

            # Universal params always pass
            if param in UNIVERSAL_PARAMS:
                if param not in seen_params:
                    filtered.append(iv)
                    seen_params.add(param or "_none")
                continue

            # Map parameter to capability key
            capability = PARAM_TO_CAPABILITY.get(param, param)

            # Check if this machine can do it
            if capability not in adjustable:
                continue  # Skip — not physically possible

            if param not in seen_params:
                filtered.append(iv)
                seen_params.add(param or "_none")

            if len(filtered) >= 4:
                break

        return filtered[:4]

    # ------------------------------------------------------------------
    # Confidence score
    # ------------------------------------------------------------------

    def _compute_confidence(
        self, causes: list[RootCause], context: BrewingContext
    ) -> float:
        if not causes:
            return 0.0
        base = causes[0].probability
        completeness = 1.0
        if not context.extraction_time_seconds:
            completeness *= 0.80
        if context.machine_type == "unknown":
            completeness *= 0.70
        return round(base * completeness, 2)

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    def _build_warnings(self, context: BrewingContext, symptoms: list[str]) -> list[str]:
        warnings = []
        if not symptoms:
            warnings.append("No symptoms detected — try describing what tastes wrong (e.g. bitter, sour, weak).")
        if context.bean_freshness_days and context.bean_freshness_days > 45:
            warnings.append("⚠️ Beans likely stale (>45 days post-roast) — try fresh beans first before adjusting parameters.")
        if context.machine_type == "unknown":
            warnings.append("Machine not identified — advice will be generic. Mention your machine model for targeted help.")
        if context.machine_type == "nespresso":
            warnings.append("ℹ️ Nespresso machines have locked extraction parameters — focus on capsule selection and machine maintenance.")
        return warnings

    # ------------------------------------------------------------------
    # Special flows for non-troubleshoot goals
    # ------------------------------------------------------------------

    def _learn_explore_result(self, context: BrewingContext) -> DiagnosticResult:
        """Lightweight result for users who want to learn, not troubleshoot."""
        starter_interventions = [
            Intervention(1, "Start with a medium roast from a reputable roaster — easier to dial in", priority="high"),
            Intervention(2, "Use filtered water at ~93°C (200°F) — water quality is 98% of the cup", "water_temp", priority="high"),
            Intervention(3, "Weigh your dose every time — consistency is the foundation of good coffee", "dose", priority="high"),
            Intervention(4, "Taste and adjust one parameter at a time — never change two things at once", priority="medium"),
        ]
        filtered = self._filter_interventions(starter_interventions, context.machine_type)
        return DiagnosticResult(
            symptoms=[],
            root_causes=[],
            intervention_plan=filtered,
            diagnostic_confidence=0.50,
            method_detected=context.machine_type,
            warnings=["ℹ️ Learning mode: showing foundational principles, not a specific fix."],
        )

    def _optimize_result(self, context: BrewingContext) -> DiagnosticResult:
        """
        Flow for users with specs who want to optimize — no symptom required.

        Strategy: inspect the provided parameters against known good ranges
        for the machine/method and flag anything out of spec.
        This is deterministic analysis, not problem diagnosis.
        """
        interventions = []
        warnings = []
        step = 1

        # --- Parameter analysis against known good ranges ---

        # Extraction time (espresso: 25-35s, moka: varies, filter: N/A)
        t = context.extraction_time_seconds
        if t is not None:
            if context.machine_type in ("super_automatic", "semi_automatic", "manual_lever"):
                if t < 22:
                    interventions.append(Intervention(
                        step, "Extraction time is short (<22s) — try 1 notch finer on grind",
                        "grind_size", "finer", "1 notch",
                        expected_result="Aim for 25-32s for balanced espresso",
                        validation_test="Taste should gain body and sweetness",
                        priority="high",
                    ))
                    step += 1
                elif t > 35:
                    interventions.append(Intervention(
                        step, "Extraction time is long (>35s) — try 1 notch coarser on grind",
                        "grind_size", "coarser", "1 notch",
                        expected_result="Aim for 25-32s to avoid over-extraction",
                        validation_test="Bitterness should decrease",
                        priority="high",
                    ))
                    step += 1
            elif context.machine_type == "v60":
                if t is not None and (t < 120 or t > 240):
                    interventions.append(Intervention(
                        step, f"V60 brew time ({t}s) outside 2-4 min range — adjust grind accordingly",
                        "grind_size", "finer" if t < 120 else "coarser", "0.5 notch",
                        expected_result="Target 2-4 minutes total brew time",
                        validation_test="Check brew time on next pour",
                        priority="medium",
                    ))
                    step += 1

        # Dose check (espresso: 14-20g, V60: 15g per 250ml typical)
        d = context.dose_grams
        if d is not None:
            if context.machine_type in ("super_automatic", "semi_automatic") and (d < 14 or d > 22):
                interventions.append(Intervention(
                    step, f"Dose {d}g outside typical espresso range (14-20g) — verify basket size",
                    "dose", priority="medium",
                    expected_result="Match dose to basket capacity",
                    validation_test="Weigh spent puck — should be ≈ dose ± 1g",
                ))
                step += 1

        # Temperature check
        temp = context.water_temp_celsius
        if temp is not None:
            if context.roast_level == "light" and temp < 92:
                interventions.append(Intervention(
                    step, f"Water temp {temp}°C may be too low for light roast — try 93-96°C",
                    "water_temp", "higher", "2-3°C",
                    expected_result="Higher temp helps extract sweetness from light roasts",
                    validation_test="Acidity should feel brighter and more balanced",
                    priority="medium",
                ))
                step += 1
            elif context.roast_level == "dark" and temp > 94:
                interventions.append(Intervention(
                    step, f"Water temp {temp}°C may be high for dark roast — try 88-92°C",
                    "water_temp", "lower", "2-3°C",
                    expected_result="Lower temp reduces harsh bitter extraction from dark roasts",
                    validation_test="Taste should feel cleaner",
                    priority="medium",
                ))
                step += 1

        # Freshness
        if context.bean_freshness_days and context.bean_freshness_days > 21:
            interventions.append(Intervention(
                step, f"Beans at {context.bean_freshness_days} days post-roast — peak window is 7-21 days",
                "freshness", priority="critical",
                expected_result="Fresher beans = more aroma, more crema, more complexity",
                validation_test="Compare same recipe with fresh beans",
            ))
            step += 1
            warnings.append(f"⚠️ Beans at {context.bean_freshness_days} days — consider sourcing fresher.")

        # If no parameters were provided at all → guide the user
        if not interventions:
            interventions.append(Intervention(
                1, "Share your current parameters (dose, time, temperature) for a specific analysis",
                priority="high",
                expected_result="With parameters, we can identify what to refine",
                validation_test="Compare before/after any adjustment",
            ))
            warnings.append(
                "ℹ️ Optimization mode: no parameters out of range, or parameters not provided. "
                "Share dose, extraction time, temperature for a targeted analysis."
            )

        filtered = self._filter_interventions(interventions, context.machine_type)

        return DiagnosticResult(
            symptoms=[],
            root_causes=[],
            intervention_plan=filtered,
            diagnostic_confidence=0.55,
            method_detected=context.machine_type,
            warnings=warnings or ["ℹ️ Optimization mode: parameters look reasonable. Fine-tuning suggestions below."],
        )

    def _general_result(self, context: BrewingContext) -> DiagnosticResult:
        """
        Minimal non-blocking result for out-of-scope questions:
        machine error codes, purchase advice, coffee science theory, recipe requests.

        The DiagnosticResult here is intentionally thin — it exists only to let
        the agent's tool loop proceed without triggering the evaluator's "blocked"
        verdict. The actual answer is produced by the agent's answer_general_question
        tool, which calls the LLM directly with appropriate scope framing.

        Confidence = 0.15 → above the 0.05 non-troubleshoot floor but clearly low,
        signalling to the agent that this is not a confident brewing diagnosis.
        """
        return DiagnosticResult(
            symptoms=[],
            root_causes=[],
            intervention_plan=[
                Intervention(
                    1,
                    "Routing to general barista knowledge — this question is outside the "
                    "core brewing troubleshooting scope.",
                    priority="medium",
                    expected_result="LLM will answer from general coffee knowledge.",
                    validation_test="",
                )
            ],
            diagnostic_confidence=0.15,
            method_detected=context.machine_type,
            warnings=[
                "ℹ️ General question detected (error code / purchase / science / recipe). "
                "Answering from barista knowledge base rather than the troubleshooting engine."
            ],
        )

    def _no_symptoms_result(self, context: BrewingContext) -> DiagnosticResult:
        return DiagnosticResult(
            symptoms=[],
            root_causes=[],
            intervention_plan=[
                Intervention(
                    1,
                    "Describe what tastes wrong (e.g. bitter, sour, weak, too strong, no aroma) "
                    "or share your parameters (dose, time, temperature) for an optimization.",
                    priority="high",
                )
            ],
            diagnostic_confidence=0.10,
            method_detected=context.machine_type,
            warnings=["Describe a symptom or share your brew parameters to get specific coaching."],
        )


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from engine.models import BrewingContext

    planner = DiagnosticPlanner()

    cases = [
        # Moka bitter — should get heat_too_high, not tamping advice
        BrewingContext(machine_type="moka", raw_problem="moka tastes burnt and bitter",
                       symptoms_detected=["bitter"], goal="troubleshoot"),
        # Nespresso bitter — should get capsule advice, NOT tamping/distribution
        BrewingContext(machine_type="nespresso", raw_problem="my nespresso tastes bitter",
                       symptoms_detected=["bitter"], goal="troubleshoot"),
        # Super-auto sour — should not suggest tamping
        BrewingContext(machine_type="super_automatic", raw_problem="espresso too sour 20 seconds",
                       extraction_time_seconds=20, symptoms_detected=["sour"], goal="troubleshoot"),
        # Bitter + sour co-occurrence
        BrewingContext(machine_type="semi_automatic", raw_problem="my espresso is both bitter and sour, very unbalanced",
                       symptoms_detected=["bitter", "sour"], goal="troubleshoot"),
        # Learn mode
        BrewingContext(machine_type="v60", raw_problem="I want to learn how to make better pour-over",
                       symptoms_detected=[], goal="learn"),
    ]

    for ctx in cases:
        result = planner.diagnose(ctx)
        print(f"\nMachine: {ctx.machine_type} | Goal: {ctx.goal} | Symptoms: {ctx.symptoms_detected}")
        print(f"  Confidence: {result.diagnostic_confidence:.0%}")
        if result.root_causes:
            print(f"  Top cause: {result.root_causes[0].hypothesis} (w={result.root_causes[0].probability:.2f})")
        print(f"  Interventions ({len(result.intervention_plan)}):")
        for iv in result.intervention_plan:
            print(f"    {iv.step}. [{iv.priority}] {iv.action}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
