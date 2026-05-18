"""
Coaching Evaluator
==================
Deterministic quality gate for coaching sessions.

Runs two passes:
1. Pre-generation: validate the diagnostic is solid enough to coach on.
2. Post-generation: validate the LLM output meets quality standards.

Fixes vs. original plan:
- Machine capability awareness: checks that interventions are
  actually possible on the user's machine.
- Separate score for retrieval quality vs. coaching quality.
- Verdict thresholds are explicit and documented.
- No forbidden phrases list (fragile regex on coffee text) -- replaced
  with actionable specificity check.
- "general" goal added to non_troubleshoot set so out-of-scope
  questions (error codes, purchase, science, recipe) are never blocked.
"""

import re
from typing import Optional

from homebarista.models import BrewingContext, DiagnosticResult
from homebarista.diagnostic_planner import MACHINE_ADJUSTABLE, PARAM_TO_CAPABILITY


# ------------------------------------------------------------------
# Thresholds (explicit and documented)
# ------------------------------------------------------------------

MIN_DIAGNOSTIC_CONFIDENCE = 0.20   # below this = too vague to coach
MIN_AVG_SEMANTIC_SCORE    = 0.25   # retrieval quality floor
MIN_COACHING_WORDS        = 120    # coaching too short = insufficient
MAX_COACHING_WORDS        = 700    # coaching too long = overwhelming
MIN_SPECIFICITY_SIGNALS   = 2      # must contain at least 2 specific measurements

# Patterns that indicate specific, actionable advice
SPECIFICITY_SIGNALS = [
    r"\d+\s*(?:deg[CF]|degrees?|celsius|fahrenheit)",  # temperatures
    r"\d+\s*(?:g(?:rams?)?|ml|seconds?|s\b)",          # measurements
    r"\d+\s*notch",                                     # grind adjustments
    r"\d+\s*(?:cran|setting)",
    r"\bfiner\b|\bcoarser\b",
    r"\bhigher\b|\blower\b|\bmore\b|\bless\b",
    r"\bincrease\b|\bdecrease\b|\breduce\b|\badd\b",
]

# Validation test indicators
VALIDATION_SIGNALS = [
    "you should notice", "you should see", "you should taste",
    "if this works", "if the fix works", "check if",
    "should improve", "should feel", "should taste",
    "look for", "notice if", "see if",
    "validation", "test by", "verify",
]

# Root cause explanation indicators
EXPLANATION_SIGNALS = [
    "because", "due to", "this happens", "the reason",
    "this causes", "which means", "result of", "caused by",
]


# ------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------

class CoachingEvaluator:
    """
    Deterministic quality gate. Never calls LLM.
    Returns structured check results and a verdict.
    """

    # ------------------------------------------------------------------
    # Pre-generation checks (on DiagnosticResult)
    # ------------------------------------------------------------------

    def evaluate_diagnostic(
        self,
        diagnostic: DiagnosticResult,
        context: BrewingContext,
        retrieval_metadata: Optional[dict] = None,
    ) -> dict:
        """
        Check if the diagnostic is solid enough to generate coaching.
        Run this BEFORE calling the LLM.

        Returns:
            {
                "checks": {check_name: bool},
                "overall_score": float,
                "warnings": [str],
                "verdict": "ready" | "review" | "blocked"
            }
        """
        checks = {}
        warnings = []

        # Non-troubleshoot goals (learn, optimize, explore, general) have empty
        # symptoms and root_causes by design -- bypass those checks to avoid blocking.
        # "general" = out-of-scope question routed to answer_general_question tool.
        non_troubleshoot = context.goal in ("learn", "optimize", "explore", "general")

        # 1. Symptoms detected -- only required for troubleshoot
        if non_troubleshoot:
            checks["symptoms_detected"] = True   # N/A for this goal
        else:
            checks["symptoms_detected"] = len(diagnostic.symptoms) >= 1

        # 2. Root causes present -- only required for troubleshoot
        if non_troubleshoot:
            checks["root_causes_present"] = True  # N/A for this goal
        else:
            checks["root_causes_present"] = len(diagnostic.root_causes) >= 1

        # 3. Intervention plan present -- always required
        checks["intervention_plan_present"] = len(diagnostic.intervention_plan) >= 1

        # 4. Confidence floor -- relaxed for non-troubleshoot
        min_conf = 0.05 if non_troubleshoot else MIN_DIAGNOSTIC_CONFIDENCE
        checks["confidence_sufficient"] = (
            diagnostic.diagnostic_confidence >= min_conf
        )

        # 5. Machine identified
        checks["machine_identified"] = context.machine_type != "unknown"
        if not checks["machine_identified"]:
            warnings.append("Machine not identified -- coaching will be generic.")

        # 6. No physically impossible interventions
        checks["interventions_feasible"] = self._check_interventions_feasible(
            diagnostic, context
        )
        if not checks["interventions_feasible"]:
            warnings.append(
                "Some interventions may not be possible on " + context.machine_type + ". "
                "Check machine capability filter."
            )

        # 7. Retrieval quality (if metadata provided)
        if retrieval_metadata:
            avg_score = retrieval_metadata.get("avg_semantic_score", 0.0)
            checks["retrieval_quality_ok"] = avg_score >= MIN_AVG_SEMANTIC_SCORE
            if not checks["retrieval_quality_ok"]:
                warnings.append(
                    "Retrieval quality low (avg_score=" + str(round(avg_score, 2)) + "). "
                    "Coaching may lack expert grounding."
                )
        else:
            checks["retrieval_quality_ok"] = True  # not checked if no metadata

        overall_score = sum(checks.values()) / len(checks)
        verdict = self._pre_verdict(checks)

        return {
            "checks": checks,
            "overall_score": round(overall_score, 2),
            "warnings": warnings,
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Post-generation checks (on coaching text)
    # ------------------------------------------------------------------

    def evaluate_coaching(
        self,
        coaching_text: str,
        diagnostic: Optional[DiagnosticResult] = None,
    ) -> dict:
        """
        Check quality of the generated coaching text.
        Run this AFTER LLM generation.

        Returns:
            {
                "checks": {check_name: bool},
                "overall_score": float,
                "warnings": [str],
                "verdict": "pass" | "warn" | "fail"
            }
        """
        checks = {}
        warnings = []
        lower = coaching_text.lower()
        words = coaching_text.split()

        # 1. Length check
        checks["appropriate_length"] = MIN_COACHING_WORDS <= len(words) <= MAX_COACHING_WORDS
        if len(words) < MIN_COACHING_WORDS:
            warnings.append("Coaching too short (" + str(len(words)) + " words < " + str(MIN_COACHING_WORDS) + " minimum).")
        if len(words) > MAX_COACHING_WORDS:
            warnings.append("Coaching too long (" + str(len(words)) + " words > " + str(MAX_COACHING_WORDS) + " maximum).")

        # 2. Specificity: at least N measurements/directions
        specificity_hits = sum(
            1 for pattern in SPECIFICITY_SIGNALS
            if re.search(pattern, coaching_text, re.IGNORECASE)
        )
        checks["specific_enough"] = specificity_hits >= MIN_SPECIFICITY_SIGNALS
        if not checks["specific_enough"]:
            warnings.append(
                "Coaching lacks specific measurements (found " + str(specificity_hits) +
                " signals, need " + str(MIN_SPECIFICITY_SIGNALS) + ")."
            )

        # 3. Validation test present
        checks["has_validation_test"] = any(sig in lower for sig in VALIDATION_SIGNALS)
        if not checks["has_validation_test"]:
            warnings.append("No validation test found. User won't know if the fix worked.")

        # 4. Explains why (not just what)
        checks["explains_root_cause"] = any(sig in lower for sig in EXPLANATION_SIGNALS)
        if not checks["explains_root_cause"]:
            warnings.append("Coaching tells user WHAT to do but not WHY. Add a brief science explanation.")

        # 5. Mentions diagnosed root cause (if diagnostic provided)
        if diagnostic and diagnostic.root_causes:
            top_hypothesis = diagnostic.root_causes[0].hypothesis.replace("-", " ").replace("_", " ")
            checks["mentions_root_cause"] = top_hypothesis in lower
            if not checks["mentions_root_cause"]:
                warnings.append(
                    "Coaching doesn't explicitly mention the primary root cause: '" + top_hypothesis + "'."
                )
        else:
            checks["mentions_root_cause"] = True  # can't check without diagnostic

        # 6. Not a wall of text (has some structure)
        checks["has_structure"] = "\n" in coaching_text or coaching_text.count(". ") >= 3
        if not checks["has_structure"]:
            warnings.append("Coaching appears to be a single block of text -- consider adding structure.")

        overall_score = sum(checks.values()) / len(checks)
        verdict = self._post_verdict(checks)

        return {
            "checks": checks,
            "overall_score": round(overall_score, 2),
            "warnings": warnings,
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_interventions_feasible(
        diagnostic: DiagnosticResult, context: BrewingContext
    ) -> bool:
        """
        Returns True if all interventions are feasible on the detected machine.
        Returns False if any intervention requires a parameter the machine can't adjust.
        """
        adjustable = MACHINE_ADJUSTABLE.get(
            context.machine_type, MACHINE_ADJUSTABLE["unknown"]
        )
        for iv in diagnostic.intervention_plan:
            param = iv.parameter
            if param is None:
                continue  # universal step, always ok
            capability = PARAM_TO_CAPABILITY.get(param, param)
            if capability not in adjustable and capability not in {"freshness", "machine_cleaning"}:
                return False
        return True

    @staticmethod
    def _pre_verdict(checks):
        """
        blocked: can't coach -- no intervention plan or confidence too low
        review:  some checks fail but coaching can proceed with caveats
        ready:   all checks pass

        NOTE: symptoms_detected and root_causes_present are set to True
        for non-troubleshoot goals (learn/optimize/explore/general) so they never
        block those flows. Only intervention_plan_present is the universal gate.
        """
        # Universal hard blocks
        if not checks.get("intervention_plan_present"):
            return "blocked"
        if not checks.get("confidence_sufficient"):
            return "blocked"

        failing = [k for k, v in checks.items() if not v]
        if not failing:
            return "ready"
        if len(failing) <= 2:
            return "review"
        return "blocked"

    @staticmethod
    def _post_verdict(checks):
        """
        fail: critical quality checks missing
        warn: minor issues, coaching usable
        pass: all checks pass
        """
        # Critical checks
        critical = ["appropriate_length", "specific_enough", "has_validation_test"]
        if any(not checks.get(c) for c in critical):
            return "fail"
        failing = [k for k, v in checks.items() if not v]
        if not failing:
            return "pass"
        return "warn"


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    from homebarista.models import BrewingContext, DiagnosticResult, RootCause, Intervention

    evaluator = CoachingEvaluator()

    ctx = BrewingContext(
        machine_type="super_automatic",
        raw_problem="DeLonghi bitter espresso",
        symptoms_detected=["bitter"],
        goal="troubleshoot",
    )
    diag = DiagnosticResult(
        symptoms=["bitter"],
        root_causes=[RootCause("over-extraction", 0.70, "bitter = over-extracted", "grind_size")],
        intervention_plan=[
            Intervention(1, "Go 1 notch coarser", "grind_size", "coarser", "1 notch"),
        ],
        diagnostic_confidence=0.70,
        method_detected="super_automatic",
    )

    pre = evaluator.evaluate_diagnostic(diag, ctx)
    print("PRE-GENERATION EVAL:")
    print("  Verdict: " + pre["verdict"] + " | Score: " + str(round(pre["overall_score"] * 100)) + "%")
    for k, v in pre["checks"].items():
        marker = "OK  " if v else "FAIL"
        print("  " + marker + " " + k)

    good_coaching = """
    Your DeLonghi Dinamica espresso tastes bitter because of over-extraction.
    This happens when the water dissolves too many compounds from the coffee grounds --
    specifically bitter chlorogenic acids that dominate when extraction runs too long
    or the grind is too fine.

    Here is how to fix it:

    1. Go 1 notch coarser on the grind setting (setting 6 to 7 if you are on 6).
       This reduces resistance and shortens contact time.
    2. Lower the temperature setting by 2 degrees C in your machine menu.
       Cooler water extracts fewer bitter compounds.

    You should notice the bitterness reduce within 2-3 shots after the adjustment.
    If the coffee starts tasting sour, you have gone too far -- go back 0.5 notches.
    """

    post = evaluator.evaluate_coaching(good_coaching, diag)
    print("\nPOST-GENERATION EVAL:")
    print("  Verdict: " + post["verdict"] + " | Score: " + str(round(post["overall_score"] * 100)) + "%")
    for k, v in post["checks"].items():
        marker = "OK  " if v else "FAIL"
        print("  " + marker + " " + k)
    if post["warnings"]:
        print("  Warnings: " + str(post["warnings"]))
