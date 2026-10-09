import json
from pathlib import Path

import yaml

from Code.v2.eval.run import load_golden, refresh_hash


def test_ci_subset_has_20_ids_with_ten_simple_ten_complex_and_two_unanswerable():
    config = yaml.safe_load(Path("Code/v2/config/eval.yaml").read_text(encoding="utf-8"))
    items = {item["id"]: item for item in load_golden()}
    selected = [items[item_id] for item_id in config["ci_subset_ids"]]
    assert len(selected) == 20
    assert sum(item["complexity"] == "simple" for item in selected) == 10
    assert sum(item["complexity"] == "complex" for item in selected) == 10
    assert sum("unanswerable" in item["tags"] for item in selected) == 2


def test_refresh_hash_updates_version_and_hash(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text('{"id":"x"}\n', encoding="utf-8")
    config = tmp_path / "eval.yaml"
    config.write_text("dataset_version: '0.1'\ngolden_sha256: old\n", encoding="utf-8")
    digest, version = refresh_hash(golden, config)
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert loaded["golden_sha256"] == digest
    assert loaded["dataset_version"] == version
    assert version != "0.1"


def test_workflow_has_push_pr_pytest_v1_guard_and_ci_eval():
    workflow = yaml.safe_load(Path(".github/workflows/v2-eval.yml").read_text(encoding="utf-8"))
    assert "push" in workflow[True] or "push" in workflow
    text = Path(".github/workflows/v2-eval.yml").read_text(encoding="utf-8")
    assert "pull_request" in text
    assert "pytest" in text
    assert "git diff" in text and "Code/query_phi.py" in text
    assert "--ci-subset" in text and "GROQ_API_KEY" in text


def test_baseline_ci_has_dataset_version_and_summary():
    baseline = json.loads(Path("Code/v2/eval/baseline-ci.json").read_text(encoding="utf-8"))
    assert baseline["dataset_version"]
    assert baseline["mode"] == "v2"
    assert "summary" in baseline
