from __future__ import annotations

from typing import Any

import httpx

from app.config import settings
from app.services.cache_service import build_cache_key, get_json, hash_payload, set_json


class TavilyError(RuntimeError):
    """Raised when a Tavily request cannot be completed."""


def _require_api_key() -> str:
    api_key = settings.tavily_api_key.strip()
    if not api_key:
        raise TavilyError("未配置 Tavily API Key，无法执行联网搜索。")
    return api_key


async def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_key = _require_api_key()
    cache_key = build_cache_key("tavily", path.strip("/"), hash_payload(payload))
    if settings.tavily_cache_enabled:
        cached = await get_json(cache_key)
        if isinstance(cached, dict):
            return cached

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    timeout = httpx.Timeout(45.0, connect=10.0)

    async with httpx.AsyncClient(base_url="https://api.tavily.com", timeout=timeout) as client:
        response = await client.post(path, json=payload, headers=headers)

    if response.status_code >= 400:
        detail = response.text.strip() or f"HTTP {response.status_code}"
        raise TavilyError(f"Tavily 请求失败：{detail}")

    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        raise TavilyError(f"Tavily 请求失败：{data['error']}")
    if settings.tavily_cache_enabled:
        await set_json(cache_key, data, ttl_seconds=settings.tavily_cache_ttl_seconds)
    return data


async def tavily_search(
    query: str,
    *,
    search_depth: str = "basic",
    topic: str = "general",
    max_results: int = 5,
    time_range: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    include_answer: bool = False,
    include_raw_content: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "search_depth": search_depth,
        "topic": topic,
        "max_results": max_results,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
    }
    if time_range:
        payload["time_range"] = time_range
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    return await _post_json("/search", payload)


async def tavily_extract(
    urls: list[str],
    *,
    extract_depth: str = "basic",
    include_images: bool = False,
) -> dict[str, Any]:
    payload = {
        "urls": urls,
        "extract_depth": extract_depth,
        "include_images": include_images,
    }
    return await _post_json("/extract", payload)


async def tavily_research(
    query: str,
    *,
    max_results: int = 5,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input": query,
        "max_results": max_results,
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    return await _post_json("/research", payload)
