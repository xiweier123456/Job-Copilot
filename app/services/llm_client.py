"""
app/services/llm_client.py
LLM 调用封装，兼容 OpenAI 格式（DeepSeek / 智谱 / 通义 等）。
"""
from __future__ import annotations

from functools import lru_cache
from openai import AsyncOpenAI

from app.config import settings


@lru_cache(maxsize=1)
def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


async def chat_completion(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> str:
    """发送对话请求，返回回复文本。"""
    client = get_llm_client()
    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
