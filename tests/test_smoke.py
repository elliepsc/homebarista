import pytest
from engine.models import BrewingContext
from engine.symptom_extractor import SymptomExtractor
from engine.diagnostic_planner import DiagnosticPlanner
from engine.coaching_evaluator import CoachingEvaluator
from ingestion.transcript_preprocessor import TranscriptPreprocessor


def test_bitter_espresso_end_to_end():
    extractor = SymptomExtractor(demo_mode=True)
    planner = DiagnosticPlanner()
    evaluator = CoachingEvaluator()

    ctx = extractor.extract("My DeLonghi Dinamica makes bitter espresso, 28 second extraction")
    assert ctx.machine_type == "super_automatic"
    assert "bitter" in ctx.symptoms_detected
    assert ctx.extraction_time_seconds == 28

    diag = planner.diagnose(ctx)
    assert diag.diagnostic_confidence > 0.0
    assert len(diag.root_causes) > 0
    assert len(diag.intervention_plan) > 0
    for iv in diag.intervention_plan:
        assert iv.parameter not in ("tamping", "distribution"), \
            f"Impossible intervention for super_automatic: {iv.action}"


def test_moka_bitter_override():
    extractor = SymptomExtractor(demo_mode=True)
    planner = DiagnosticPlanner()
    ctx = extractor.extract("My moka pot makes burnt bitter coffee")
    diag = planner.diagnose(ctx)
    assert diag.root_causes[0].hypothesis == "heat_too_high"


def test_nespresso_no_impossible_interventions():
    extractor = SymptomExtractor(demo_mode=True)
    planner = DiagnosticPlanner()
    ctx = extractor.extract("My Nespresso makes bitter coffee")
    diag = planner.diagnose(ctx)
    for iv in diag.intervention_plan:
        assert iv.parameter not in ("tamping", "distribution", "pressure"), \
            f"Impossible for Nespresso: {iv.action}"


def test_symptom_extraction_12_types():
    extractor = SymptomExtractor(demo_mode=True)
    cases = [
        ("coffee too strong overwhelming", "too_strong"),
        ("mouth feels dry astringent", "astringent"),
        ("no aroma flat smell", "flat_no_aroma"),
        ("inconsistent results every time", "inconsistent"),
        ("no flow blocked choked", "too_slow_extraction"),
        ("too fast gushing watery shot", "too_fast_extraction"),
    ]
    for text, expected_symptom in cases:
        ctx = extractor.extract(text)
        assert expected_symptom in ctx.symptoms_detected, \
            f"Expected '{expected_symptom}' in symptoms for: '{text}'"


def test_preprocessor_removes_timestamps():
    preprocessor = TranscriptPreprocessor()
    raw = "[00:03:45] so the grind size matters [01:22:00] especially for espresso"
    cleaned = preprocessor.clean(raw)
    assert "[" not in cleaned
    assert "grind size matters" in cleaned


def test_chunk_ids_are_stable():
    from ingestion.embedder import Embedder
    embedder = Embedder()
    doc = {"source_id": "test_001", "title": "T", "channel": "C",
           "url": "u", "domain": "d", "method": "m", "difficulty": "b"}
    transcript = "The grind size is the most important variable in espresso. " * 20
    chunks1 = embedder.chunk_transcript(transcript, doc)
    chunks2 = embedder.chunk_transcript(transcript, doc)
    assert [c["chunk_id"] for c in chunks1] == [c["chunk_id"] for c in chunks2], \
        "Chunk IDs must be stable across identical inputs"
