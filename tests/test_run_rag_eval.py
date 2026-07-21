"""Tests for evals.run_rag_eval helpers that don't require a live LLM."""

from unittest.mock import MagicMock, patch

from evals.run_rag_eval import judge_style


def test_judge_style_passes_reasoning_effort():
    """C9: the judge call must cap reasoning tokens like the classifier/eval sites."""
    mock_client = MagicMock()
    mock_client.create.return_value = MagicMock(
        text='{"specificity": 4, "science": 4, "actionability": 4, "completeness": 4}'
    )

    with patch("engine.llm_client.LLMClient", return_value=mock_client), \
         patch("evals.run_rag_eval.time.sleep", return_value=None):
        result = judge_style(["some coaching text"])

    assert mock_client.create.call_args.kwargs["reasoning_effort"] == "low"
    assert result["n_judged"] == 1
    assert result["parse_failures"] == 0
