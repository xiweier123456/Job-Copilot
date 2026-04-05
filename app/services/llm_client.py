"""
app/services/llm_client.py
LLM 调用封装，兼容 OpenAI 格式（DeepSeek / 智谱 / 通义 等）。
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from openai import AsyncOpenAI

from app.config import settings


@lru_cache(maxsize=1)
def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.service_llm_api_key,
        base_url=settings.service_llm_base_url,
    )


async def chat_completion(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> str:
    """发送对话请求，返回回复文本。"""
    client = get_llm_client()
    response = await client.chat.completions.create(
        model=settings.service_llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def parse_json_response(text: str) -> dict:
    """从 LLM 回复中提取 JSON，容错处理。"""
    cleaned = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}


async def chat_json_completion(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> dict:
    """发送对话请求并解析 JSON。"""
    reply = await chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return parse_json_response(reply)
