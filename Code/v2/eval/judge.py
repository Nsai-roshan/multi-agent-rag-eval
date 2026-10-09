"""Single, mode-independent claim judge for evaluation."""

from __future__ import annotations

from typing import Any

from ..costing import CostLedger
from ..verifier import Verifier


def judge_answer(
    answer_text: str,
    chunks: Any,
    *,
    verifier: Verifier | None = None,
    ledger: CostLedger | None = None,
) -> dict[str, Any]:
    """Judge an answer with the Phase-4 decomposition and one batched call.

    The evaluator intentionally does not consume a pipeline's internal
    verdicts. Both v1 and v2 answers enter this same path.
    """
    active_ledger = ledger or CostLedger()
    if not chunks:
        return {
            "claims": [],
            "verdicts": [],
            "passed": True,
            "cost_usd": active_ledger.usd_total,
            "ledger": active_ledger.calls,
        }
    active_verifier = verifier or Verifier(ledger=active_ledger)
    report = active_verifier.verify({"answer": answer_text, "claims": []}, chunks)
    return {
        "claims": report.claims,
        "verdicts": report.verdicts,
        "passed": report.passed,
        "cost_usd": report.cost_usd,
        "ledger": report.ledger.calls,
    }
