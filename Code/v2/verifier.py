"""Two-stage claim decomposition and batched entailment verification."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .costing import CostLedger
from .router import CONFIG_PATH, _load_config, _make_client, _provider_config, _resolve_env


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIMS_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "verifier_claims.txt"
JUDGE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "verifier_judge.txt"
load_dotenv(REPO_ROOT / ".env")


@dataclass
class VerifyReport:
    claims: list[dict[str, Any]]
    verdicts: list[dict[str, Any]]
    passed: bool
    ledger: CostLedger

    @property
    def cost_usd(self) -> float:
        return self.ledger.usd_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": self.claims,
            "verdicts": self.verdicts,
            "passed": self.passed,
            "cost_usd": self.cost_usd,
        }


def _parse_json(content: str) -> Any:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)


def _as_list(payload: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return [item for item in payload[key] if isinstance(item, dict)]
    return []


class Verifier:
    def __init__(
        self,
        *,
        client: Any | None = None,
        provider: str | None = None,
        config_path: str | Path = CONFIG_PATH,
        claims_prompt_path: str | Path = CLAIMS_PROMPT_PATH,
        judge_prompt_path: str | Path = JUDGE_PROMPT_PATH,
        ledger: CostLedger | None = None,
    ) -> None:
        self.config = _load_config(config_path)
        configured = _resolve_env(self.config.get("LLM_PROVIDER", "groq"))
        self.provider = provider or os.getenv("LLM_PROVIDER") or configured
        _provider_config(self.config, self.provider)
        model_set = self.config.get("model_sets", {}).get(self.provider, {})
        prefix = self.provider.upper()
        self.verifier_model = os.getenv(f"{prefix}_VERIFIER_MODEL") or model_set.get(
            "verifier_model", self.config.get("verifier_model")
        )
        self.client = client or _make_client(self.config, self.provider)
        self.claims_prompt = Path(claims_prompt_path).read_text(encoding="utf-8")
        self.judge_prompt = Path(judge_prompt_path).read_text(encoding="utf-8")
        self.ledger = ledger or CostLedger()

    @staticmethod
    def _draft_fields(draft: Any) -> tuple[str, list[dict[str, Any]]]:
        if isinstance(draft, dict):
            return str(draft.get("answer", "")), list(draft.get("claims", []))
        return str(getattr(draft, "answer", draft)), list(getattr(draft, "claims", []))

    @staticmethod
    def _chunk_context(chunks: Any) -> tuple[str, list[str]]:
        allowed = [item["chunk_id"] for item in chunks]
        lines = [f"{item['chunk_id']}: {item.get('text', '')}" for item in chunks]
        return "\n".join(lines), allowed

    def _call_json(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.verifier_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=120,
        )
        return response.choices[0].message.content

    def _decompose(
        self, answer: str, draft_claims: list[dict[str, Any]], chunks: Any, allowed: list[str]
    ) -> list[dict[str, Any]]:
        prompt = self.claims_prompt.format(
            allowed_chunk_ids=json.dumps(allowed),
            draft_answer=answer,
            draft_claims=json.dumps(draft_claims),
        )
        raw = self._call_json(prompt)
        self.ledger.record_text(self.verifier_model, prompt, raw)
        claims = []
        for index, claim in enumerate(_as_list(_parse_json(raw), "claims"), start=1):
            citation_ids = [cid for cid in claim.get("citation_ids", []) if cid in allowed]
            claims.append(
                {
                    "claim_id": str(claim.get("claim_id") or f"c{index}"),
                    "claim_text": str(claim.get("claim_text", "")),
                    "citation_ids": citation_ids,
                }
            )
        return claims

    def _judge(
        self, claims: list[dict[str, Any]], chunk_context: str
    ) -> list[dict[str, Any]]:
        prompt = self.judge_prompt.format(
            claims=json.dumps(claims),
            chunks=chunk_context,
        )
        raw = self._call_json(prompt)
        self.ledger.record_text(self.verifier_model, prompt, raw)
        return _as_list(_parse_json(raw), "verdicts")

    def verify(self, draft: Any, chunks: Any) -> VerifyReport:
        answer, draft_claims = self._draft_fields(draft)
        chunk_context, allowed = self._chunk_context(chunks)
        claims = self._decompose(answer, draft_claims, chunks, allowed)
        raw_verdicts = self._judge(claims, chunk_context)
        by_id = {str(item.get("claim_id")): item for item in raw_verdicts}
        verdicts: list[dict[str, Any]] = []
        allowed_set = set(allowed)
        for claim in claims:
            item = by_id.get(claim["claim_id"], {})
            verdict = str(item.get("verdict", "UNSUPPORTED")).upper()
            evidence = [cid for cid in item.get("evidence_chunk_ids", []) if cid in allowed_set]
            if verdict not in {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}:
                verdict = "UNSUPPORTED"
            if not claim["citation_ids"]:
                verdict = "UNSUPPORTED"
            if verdict == "SUPPORTED" and not set(evidence).intersection(claim["citation_ids"]):
                verdict = "UNSUPPORTED"
            verdicts.append(
                {
                    "claim_id": claim["claim_id"],
                    "verdict": verdict,
                    "evidence_chunk_ids": evidence,
                }
            )
        passed = not any(item["verdict"] == "UNSUPPORTED" for item in verdicts)
        LOGGER.info(
            "verifier claims=%d supported=%d partial=%d unsupported=%d passed=%s",
            len(verdicts),
            sum(item["verdict"] == "SUPPORTED" for item in verdicts),
            sum(item["verdict"] == "PARTIAL" for item in verdicts),
            sum(item["verdict"] == "UNSUPPORTED" for item in verdicts),
            passed,
        )
        return VerifyReport(claims, verdicts, passed, self.ledger)


def verify(draft: Any, chunks: Any, **kwargs: Any) -> VerifyReport:
    return Verifier(**kwargs).verify(draft, chunks)
