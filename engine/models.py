"""
Models — Shared Data Structures
================================
All dataclasses used across HomeBarista components.

Design rules:
- All fields have defaults so components can construct partial objects.
- `symptoms_detected` is the critical bridge between SymptomExtractor
  and DiagnosticPlanner — always a list, never None.
- `goal` drives routing in DiagnosticPlanner and CoachingEvaluator:
  "troubleshoot" | "optimize" | "learn" | "explore" | "general"
- Heuristic weights stored as `probability` in RootCause for model compat,
  but documented as expert heuristics — not empirical probabilities.
"""

from dataclasses import dataclass, field
from typing import Optional


# ------------------------------------------------------------------
# Brewing context — output of SymptomExtractor
# ------------------------------------------------------------------

@dataclass
class BrewingContext:
    """Structured representation of the user's brewing situation."""

    # Machine
    machine_type:  str = "unknown"   # super_automatic | moka | v60 | aeropress | ...
    machine_model: Optional[str] = None  # e.g. "DeLonghi Dinamica"

    # Bean
    bean_origin:         Optional[str] = None   # e.g. "Ethiopian"
    roast_level:         Optional[str] = None   # light | medium | dark
    bean_freshness_days: Optional[int] = None   # days since roast

    # Brew parameters (extracted from user input)
    grind_size:               Optional[str]   = None  # "fine", "setting 7", etc.
    dose_grams:               Optional[float] = None
    water_temp_celsius:       Optional[float] = None
    extraction_time_seconds:  Optional[int]   = None
    water_ratio:              Optional[str]   = None  # e.g. "1:15"

    # Raw input & intent
    raw_problem:       str       = ""
    goal:              str       = "troubleshoot"  # troubleshoot|optimize|learn|explore|general
    symptoms_detected: list[str] = field(default_factory=list)
    # ^ Critical bridge field: SymptomExtractor → DiagnosticPlanner
    # Always a list; empty list means "no symptoms detected" (not None).


# ------------------------------------------------------------------
# Diagnostic output — output of DiagnosticPlanner
# ------------------------------------------------------------------

@dataclass
class RootCause:
    """A hypothesis about why the coffee problem is occurring."""
    hypothesis:         str            # e.g. "over-extraction"
    probability:        float          # heuristic weight 0.0–1.0 (NOT a true probability)
    evidence:           str = ""       # reasoning / extraction science rationale
    parameter_affected: Optional[str] = None  # grind_size | water_temp | dose | ...


@dataclass
class Intervention:
    """A single actionable fix step for the user."""
    step:            int
    action:          str
    parameter:       Optional[str]  = None   # which variable to change
    direction:       Optional[str]  = None   # "finer" | "coarser" | "higher" | "lower"
    magnitude:       Optional[str]  = None   # "1 notch" | "2°C" | "3-5s"
    expected_result: str            = ""
    validation_test: str            = ""
    priority:        str            = "high"  # critical | high | medium | low


@dataclass
class DiagnosticResult:
    """Full diagnostic output for a user's coffee problem."""
    symptoms:              list[str]        = field(default_factory=list)
    root_causes:           list[RootCause]  = field(default_factory=list)
    intervention_plan:     list[Intervention] = field(default_factory=list)
    diagnostic_confidence: float            = 0.0
    method_detected:       str              = "unknown"
    warnings:              list[str]        = field(default_factory=list)


# ------------------------------------------------------------------
# Session — top-level log record
# ------------------------------------------------------------------

@dataclass
class CoachingSession:
    """Full record of a coaching session (used for logging & eval)."""
    session_id:         str
    raw_problem:        str
    context:            Optional[BrewingContext]  = None
    diagnostic:         Optional[DiagnosticResult] = None
    coaching_text:      str                       = ""
    evaluation:         dict                      = field(default_factory=dict)
    tool_call_log:      list[dict]                = field(default_factory=list)
    iterations:         int                       = 0
    status:             str                       = "pending"  # coaching | clarification_needed | error
