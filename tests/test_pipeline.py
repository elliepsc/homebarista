"""Tests for pipeline.pipeline linear-mode generation."""

from unittest.mock import MagicMock, patch

from engine.models import BrewingContext, DiagnosticResult, Intervention, RootCause
from pipeline.pipeline import _generate_coaching_linear


def test_generate_coaching_linear_passes_reasoning_effort():
    """C9: the generation call must cap reasoning tokens like the classifier/eval sites."""
    ctx = BrewingContext(machine_type="super_automatic", raw_problem="bitter espresso")
    diagnostic = DiagnosticResult(
        symptoms=["bitter"],
        root_causes=[RootCause(hypothesis="over_extraction", probability=0.8)],
        intervention_plan=[Intervention(step=1, action="grind coarser")],
    )
    mock_client = MagicMock()
    mock_client.create.return_value = MagicMock(text="coaching text")

    with patch("engine.llm_client.LLMClient", return_value=mock_client):
        result = _generate_coaching_linear(
            ctx, diagnostic, chunks=[], style="concise", demo_mode=False
        )

    assert result == "coaching text"
    assert mock_client.create.call_args.kwargs["reasoning_effort"] == "low"
