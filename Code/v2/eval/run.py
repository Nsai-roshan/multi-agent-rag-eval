"""Sequential golden-set evaluator for the frozen v1/v2 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from ..baseline import baseline_answer
from ..pipeline import _result_from_state
from ..graph import Pipeline
from .judge import judge_answer

ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = ROOT / "Code" / "v2" / "eval" / "golden.jsonl"
EVAL_CONFIG = ROOT / "Code" / "v2" / "config" / "eval.yaml"
SCHEMA_PATH = Path(__file__).resolve().parent / "report_schema.json"


def load_golden(path: str | Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_golden_hash(path: str | Path = GOLDEN_PATH, *, expected_hash: str | None = None, dataset_version: str | None = None) -> str:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_hash is None:
        config = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8")) or {}
        expected_hash = config.get("golden_sha256")
        dataset_version = dataset_version or config.get("dataset_version")
    if not expected_hash or expected_hash == "PLACEHOLDER" or digest != expected_hash:
        raise ValueError(f"golden dataset hash mismatch for dataset_version={dataset_version or 'unknown'}: expected {expected_hash}, got {digest}")
    return digest


def _is_429(error: Exception) -> bool:
    response = getattr(error, "response", None)
    return getattr(error, "status_code", None) == 429 or getattr(response, "status_code", None) == 429 or "429" in str(error)


MAX_429_RETRIES = 5


def _with_backoff(fn: Callable[[], Any], *, retries: int = MAX_429_RETRIES, base_seconds: float = 1.0) -> Any:
    retries = min(max(0, retries), MAX_429_RETRIES)
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as error:
            if not _is_429(error):
                raise
            if attempt >= retries:
                raise RuntimeError(
                    f"API request failed after {retries} retries due to HTTP 429 rate limiting"
                ) from error
            time.sleep(base_seconds * (2**attempt))


def _answer_v1(item: dict[str, Any]) -> tuple[str, list[str], list[dict[str, Any]], float, float, str | None, bool]:
    start = time.perf_counter()
    result = baseline_answer(item["question"])
    return result.answer, result.citations, result.chunks, result.cost_usd, (time.perf_counter() - start) * 1000, None, False


def _answer_v2(item: dict[str, Any], pipeline: Pipeline | None = None) -> tuple[str, list[str], list[dict[str, Any]], float, float, str | None, bool]:
    start = time.perf_counter()
    active = pipeline or Pipeline()
    state = active.run(item["question"])
    result = _result_from_state(state)
    return result.answer, result.citations, list(state.get("chunks", [])), result.cost_usd, state.get("latency_ms", (time.perf_counter() - start) * 1000), state.get("tier"), result.abstained


def _metrics(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    answerable = [r for r in rows if "unanswerable" not in r["tags"]]
    scored = [r for r in answerable if not (r["abstained"] and not r["claims"])]
    claims = [claim for row in scored for claim in row["judge"]["verdicts"]]
    supported = sum(c["verdict"] == "SUPPORTED" for c in claims)
    unsupported = sum(c["verdict"] == "UNSUPPORTED" for c in claims)
    cited = [cid for row in scored for cid in row["citations"]]
    expected = [cid for row in scored for cid in row["supporting_doc_ids"]]
    correct_cites = sum(cid in row["supporting_doc_ids"] for row in scored for cid in row["citations"])
    recall_hits = sum(cid in row["citations"] for row in scored for cid in row["supporting_doc_ids"])
    latencies = sorted(r["latency_ms"] for r in rows)
    def percentile(p: float) -> float:
        if not latencies: return 0.0
        pos = (len(latencies) - 1) * p
        low, high = int(pos), min(int(pos) + 1, len(latencies) - 1)
        return latencies[low] + (latencies[high] - latencies[low]) * (pos - low)
    return {
        "question_count": len(rows),
        "hallucination_rate": unsupported / len(claims) if claims else 0.0,
        "groundedness": supported / len(claims) if claims else 0.0,
        "citation_precision": correct_cites / len(cited) if cited else 0.0,
        "citation_recall": recall_hits / len(expected) if expected else 0.0,
        "avg_cost_usd": statistics.mean(r["cost_usd"] for r in rows) if rows else 0.0,
        "router_accuracy": sum(r["router_label"] == r["complexity"] for r in rows) / len(rows) if mode == "v2" and rows else None,
        "abstention_rate_unanswerable": sum(r["abstained"] for r in rows if "unanswerable" in r["tags"]) / max(1, sum("unanswerable" in r["tags"] for r in rows)),
        "p50_latency_ms": percentile(0.50),
        "p95_latency_ms": percentile(0.95),
    }


def validate_report(path: str | Path) -> bool:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        from jsonschema import validate
        validate(report, json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    except ImportError:
        if not {"schema_version", "dataset_version", "mode", "rows", "summary"} <= report.keys():
            raise ValueError("report schema validation failed")
    return True


def run_evaluation(mode: str, *, limit: int | None = None, delay: float | None = None, golden_path: str | Path = GOLDEN_PATH, pipeline: Pipeline | None = None, answer_v1: Callable = _answer_v1, answer_v2: Callable = _answer_v2, judge: Callable = judge_answer) -> tuple[dict[str, Any], Path]:
    if mode not in {"v1", "v2"}: raise ValueError("mode must be v1 or v2")
    validate_golden_hash(golden_path)
    config = yaml.safe_load(EVAL_CONFIG.read_text(encoding="utf-8")) or {}
    items = load_golden(golden_path)[:limit]
    rows = []
    active_pipeline = pipeline or (Pipeline() if mode == "v2" and answer_v2 is _answer_v2 else None)
    pause = float(os.getenv("EVAL_DELAY_SECONDS", "0.0")) if delay is None else delay
    for item in items:
        start = time.perf_counter()
        fn = answer_v1 if mode == "v1" else lambda x: answer_v2(x, active_pipeline)
        answer_text, citations, chunks, cost, latency, router_label, abstained = _with_backoff(lambda: fn(item))
        judged = _with_backoff(lambda: judge(answer_text, chunks))
        rows.append({"id": item["id"], "question": item["question"], "complexity": item["complexity"], "tags": item["tags"], "supporting_doc_ids": item["supporting_doc_ids"], "answer": answer_text, "citations": citations, "claims": judged["claims"], "judge": judged, "cost_usd": cost, "latency_ms": latency or (time.perf_counter() - start) * 1000, "abstained": abstained or (not judged["passed"] and not judged["verdicts"]), "router_label": router_label})
        if pause > 0: time.sleep(pause)
    report = {"schema_version": "1.0", "dataset_version": config.get("dataset_version", "unknown"), "golden_sha256": validate_golden_hash(golden_path), "mode": mode, "rows": rows, "summary": _metrics(rows, mode)}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / "Code" / "v2" / "eval" / "reports" / timestamp
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    validate_report(report_path)
    summary = report["summary"]
    md = "# Evaluation report\n\n| Metric | Value |\n|---|---:|\n" + "\n".join(f"| {key} | {value} |" for key, value in summary.items()) + "\n"
    (out / "report.md").write_text(md, encoding="utf-8")
    return report, report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["v1", "v2"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--delay", type=float, default=None)
    args = parser.parse_args()
    report, path = run_evaluation(args.mode, limit=args.limit, delay=args.delay)
    print(json.dumps({"report": str(path), "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
