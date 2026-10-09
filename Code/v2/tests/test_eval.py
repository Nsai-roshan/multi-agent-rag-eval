import json
from pathlib import Path

import pytest

from Code.v2.eval.compare import compare_reports, score_change
from Code.v2.eval.run import load_golden, validate_golden_hash


def test_golden_set_has_80_unverified_items_with_required_fields():
    items = load_golden()
    assert len(items) == 80
    assert all(isinstance(item["human_verified"], bool) for item in items)
    assert all(set(("id", "question", "expected_answer", "supporting_doc_ids", "complexity", "tags")) <= item.keys() for item in items)
    assert {item["complexity"] for item in items} <= {"simple", "complex"}


def test_hash_mismatch_refuses_dataset(tmp_path):
    copied = tmp_path / "golden.jsonl"
    copied.write_text(Path("Code/v2/eval/golden.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_golden_hash(copied, expected_hash="0" * 64, dataset_version="1")


def test_compare_score_change_math():
    assert score_change(0.5, 0.25) == pytest.approx(0.5)
    assert score_change(0.0, 0.25) is None
    result = compare_reports({"summary": {"hallucination_rate": 0.5, "avg_cost_usd": 0.02}}, {"summary": {"hallucination_rate": 0.25, "avg_cost_usd": 0.01}})
    assert result["SC-1"]["relative_reduction"] == pytest.approx(0.5)
    assert result["SC-2"]["relative_reduction"] == pytest.approx(0.5)


def test_report_schema_accepts_smoke_shape(tmp_path):
    from Code.v2.eval.run import validate_report

    report = {"schema_version": "1.0", "dataset_version": "1", "mode": "v1", "rows": [], "summary": {}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_report(path) is True
