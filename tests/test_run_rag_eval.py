"""Tests for evals.run_rag_eval helpers that don't require a live LLM."""

import json
from unittest.mock import MagicMock, patch

import evals.run_rag_eval as rag_eval
from evals.run_rag_eval import build_coaching_dataset, judge_style


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


def test_build_coaching_dataset_filters_out_no_symptom_queries(tmp_path, monkeypatch):
    """C10: factual/no-symptom queries would hit pipeline.py's "Could not
    diagnose" guard and inflate the error count — filter them out offline,
    no LLM call, before the live coaching eval ever runs."""
    raw_dataset = [
        {
            "query_id": "q1",
            "synthetic_query": "My DeLonghi espresso is bitter and burnt tasting, 28 second shot",
            "relevant_chunk_ids": ["a"],
            "machine_type": "espresso",
            "domain": "troubleshooting",
        },
        {
            "query_id": "q2",
            "synthetic_query": (
                "What is the exact origin and processing details of the coffee I "
                "bought - which farm, altitude, and processing method was used?"
            ),
            "relevant_chunk_ids": ["b"],
            "machine_type": "espresso",
            "domain": "origin",
        },
    ]
    dataset_path = tmp_path / "eval_dataset.json"
    coaching_path = tmp_path / "eval_dataset_coaching.json"
    dataset_path.write_text(json.dumps(raw_dataset), encoding="utf-8")

    monkeypatch.setattr(rag_eval, "DATASET", dataset_path)
    monkeypatch.setattr(rag_eval, "COACHING_DATASET", coaching_path)

    build_coaching_dataset()

    kept = json.loads(coaching_path.read_text(encoding="utf-8"))
    kept_ids = {item["query_id"] for item in kept}
    assert kept_ids == {"q1"}
