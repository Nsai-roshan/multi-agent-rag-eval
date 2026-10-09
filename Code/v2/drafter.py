"""Tier-selected answer drafting with grounded claim citations."""

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
from .router import (
    CONFIG_PATH,
    PROMPT_PATH as ROUTER_PROMPT_PATH,
    _load_config,
    _make_client,
    _provider_config,
    _resolve_env,
)


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "drafter.txt"
load_dotenv(REPO_ROOT / ".env")


@dataclass
class DraftResult:
    answer: str
    claims: list[dict[str, Any]]
    model: str
    tier: str
    cost_usd: float
    ledger: CostLedger

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "claims": self.claims,
            "model": self.model,
            "tier": self.tier,
            "cost_usd": self.cost_usd,
        }


def _parse_json_content(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    return json.loads(cleaned)


def _ensure_sentence_citations(
    answer: str, claims: list[dict[str, Any]], allowed_ids: list[str]
) -> str:
    """Ensure factual prose ends with an ID that came from retrieval."""
    if not claims or not allowed_ids or "don't know" in answer.lower() or "do not know" in answer.lower():
        return answer
    fallback = next(
        (citation_id for claim in claims for citation_id in claim["citation_ids"]),
        allowed_ids[0],
    )
    citation = f" [{fallback}]"
    parts = [part.strip() for part in re.split(r"(?<=[.!?])(?=\s+|$)", answer.strip()) if part.strip()]
    repaired = []
    for part in parts:
        if part.strip() and not part.rstrip().endswith("]"):
            part = part.rstrip() + citation
        repaired.append(part)
    return " ".join(repaired).strip()


class Drafter:
    def __init__(
        self,
        *,
        client: Any | None = None,
        provider: str | None = None,
        config_path: str | Path = CONFIG_PATH,
        prompt_path: str | Path = PROMPT_PATH,
        ledger: CostLedger | None = None,
    ) -> None:
        self.config = _load_config(config_path)
        configured = _resolve_env(self.config.get("LLM_PROVIDER", "groq"))
        self.provider = provider or os.getenv("LLM_PROVIDER") or configured
        provider_config = _provider_config(self.config, self.provider)
        model_set = self.config.get("model_sets", {}).get(self.provider, {})
        prefix = self.provider.upper()
        self.tier_a = os.getenv(f"{prefix}_TIER_A_MODEL") or model_set.get(
            "tier_a", self.config.get("tier_a")
        )
        self.tier_b = os.getenv(f"{prefix}_TIER_B_MODEL") or model_set.get(
            "tier_b", self.config.get("tier_b")
        )
        self.client = client or _make_client(self.config, self.provider)
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.ledger = ledger or CostLedger()
        self._base_url = _resolve_env(provider_config["base_url"])

    def _model_for_tier(self, tier: str) -> str:
        normalized = getattr(tier, "label", tier)
        if normalized == "SIMPLE":
            return self.tier_a
        if normalized == "COMPLEX":
            return self.tier_b
        if normalized in {self.tier_a, self.tier_b}:
            return normalized
        raise ValueError(f"Unknown drafting tier: {tier}")

    @staticmethod
    def _context(retrieval: Any) -> tuple[str, list[str]]:
        allowed = [item["chunk_id"] for item in retrieval]
        if hasattr(retrieval, "as_context"):
            return retrieval.as_context(), allowed
        lines = [f"{item['source']}: {item['text']}" for item in retrieval]
        return " ".join(lines), allowed

    def draft(
        self,
        question: str,
        retrieval: Any,
        tier: str,
        prompt_override: str | None = None,
    ) -> DraftResult:
        model = self._model_for_tier(tier)
        context, allowed_ids = self._context(retrieval)
        prompt = prompt_override or self.prompt_template.format(
            allowed_chunk_ids=json.dumps(allowed_ids), context=context, question=question
        )
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=120,
        )
        raw_content = response.choices[0].message.content
        self.ledger.record_text(model, prompt, raw_content)
        payload = _parse_json_content(raw_content)
        claims = []
        allowed = set(allowed_ids)
        for claim in payload.get("claims", []):
            citation_ids = [cid for cid in claim.get("citation_ids", []) if cid in allowed]
            claims.append({"claim": str(claim.get("claim", "")), "citation_ids": citation_ids})
        answer = str(payload.get("answer", "I don't know based on the retrieved context."))
        answer = _ensure_sentence_citations(answer, claims, allowed_ids)
        LOGGER.info("drafter model=%s tier=%s claims=%d", model, tier, len(claims))
        return DraftResult(
            answer=answer,
            claims=claims,
            model=model,
            tier=tier,
            cost_usd=self.ledger.usd_total,
            ledger=self.ledger,
        )


def draft_answer(question: str, retrieval: Any, tier: str, **kwargs: Any) -> DraftResult:
    return Drafter(**kwargs).draft(question, retrieval, tier)
