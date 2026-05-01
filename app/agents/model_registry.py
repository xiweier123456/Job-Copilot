from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentModelSpec:
    provider: str
    label: str
    model: str
    api_key: str
    base_url: str
    is_default: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)


def _available_agent_model_specs() -> dict[str, AgentModelSpec]:
    return {
        "minimax": AgentModelSpec(
            provider="minimax",
            label="MiniMax Agent",
            model=settings.agent_llm_model,
            api_key=settings.agent_llm_api_key,
            base_url=settings.agent_llm_base_url,
            is_default=True,
        ),
        "deepseek": AgentModelSpec(
            provider="deepseek",
            label="DeepSeek Agent",
            model=settings.service_llm_model,
            api_key=settings.service_llm_api_key,
            base_url=settings.service_llm_base_url,
        ),
    }


def resolve_agent_model_spec(provider: str | None = None) -> AgentModelSpec:
    specs = _available_agent_model_specs()
    key = (provider or "minimax").strip().lower()
    spec = specs.get(key) or specs["minimax"]
    if spec.configured:
        return spec

    for fallback in specs.values():
        if fallback.configured:
            logger.warning("Agent model %s is not configured; falling back to %s", spec.provider, fallback.provider)
            return fallback
    return spec


def get_agent_model_options() -> list[dict[str, Any]]:
    return [
        {
            "provider": spec.provider,
            "label": spec.label,
            "model": spec.model,
            "base_url": spec.base_url,
            "configured": spec.configured,
            "default": spec.is_default,
        }
        for spec in _available_agent_model_specs().values()
    ]
