"""Few-shot SIMPLE/COMPLEX query routing."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .costing import CostLedger


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "models.yaml"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "router.txt"
load_dotenv(REPO_ROOT / ".env")


def _resolve_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        body = value[2:-1]
        if ":-" in body:
            name, default = body.split(":-", 1)
            return os.getenv(name, default)
        return os.getenv(body, "")
    return value


def _load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _provider_config(config: dict[str, Any], provider: str) -> dict[str, Any]:
    selected = config.get("providers", {}).get(provider)
    if selected is None:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    return selected


def _make_client(config: dict[str, Any], provider: str):
    from openai import OpenAI

    provider_config = _provider_config(config, provider)
    base_url = _resolve_env(provider_config["base_url"]).rstrip("/")
    api_key = os.getenv(provider_config.get("api_key_env") or "") or "ollama"
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)


@dataclass(frozen=True)
class RouterDecision:
    label: str
    confidence: float
    tier: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "tier": self.tier,
            "model": self.model,
        }


class Router:
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
        _provider_config(self.config, self.provider)
        model_set = self.config.get("model_sets", {}).get(self.provider, {})
        prefix = self.provider.upper()
        self.router_model = os.getenv(f"{prefix}_ROUTER_MODEL") or model_set.get(
            "router_model", self.config.get("router_model")
        )
        self.tier_a = os.getenv(f"{prefix}_TIER_A_MODEL") or model_set.get(
            "tier_a", self.config.get("tier_a")
        )
        self.tier_b = os.getenv(f"{prefix}_TIER_B_MODEL") or model_set.get(
            "tier_b", self.config.get("tier_b")
        )
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.client = client or _make_client(self.config, self.provider)
        self.ledger = ledger

    def classify(self, question: str) -> RouterDecision:
        prompt = self.prompt_template.format(question=question)
        response = self.client.chat.completions.create(
            model=self.router_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
            timeout=120,
        )
        payload = json.loads(response.choices[0].message.content)
        if self.ledger is not None:
            self.ledger.record_text(self.router_model, prompt, response.choices[0].message.content)
        label = str(payload.get("label", "COMPLEX")).upper()
        if label not in {"SIMPLE", "COMPLEX"}:
            raise ValueError(f"Invalid router label: {label}")
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
        tier = self.tier_a if label == "SIMPLE" else self.tier_b
        LOGGER.info(
            "router label=%s confidence=%.3f chosen_tier=%s model=%s",
            label,
            confidence,
            "tier_a" if label == "SIMPLE" else "tier_b",
            tier,
        )
        return RouterDecision(label, confidence, tier, self.router_model)
