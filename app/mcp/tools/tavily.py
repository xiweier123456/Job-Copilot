"""
app/mcp/tools/tavily.py
MCP tools：Tavily 联网搜索 / 研究 / 抽取
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.services.tavily_client import (
    TavilyError,
    tavily_extract as _tavily_extract,
    tavily_research as _tavily_research,
    tavily_search as _tavily_search,
)


def _split_csv(value: str) -> list[str] | None:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _build_search_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    output = []
    for item in items[:limit]:
        content = (item.get("content") or "").strip()
        output.append(
            {
                "title": item.get("title") or "无标题",
                "url": item.get("url") or "无链接",
                "score": item.get("score"),
                "snippet": content[:240] + ("..." if len(content) > 240 else ""),
            }
        )
    return output


async def tavily_search(
    query: str,
    include_domains: str = "",
    exclude_domains: str = "",
    max_results: int = 5,
    topic: str = "general",
    time_range: str = "",
) -> dict:
    try:
        result = await _tavily_search(
            query=query,
            max_results=max_results,
            topic=topic,
            time_range=time_range or None,
            include_domains=_split_csv(include_domains),
            exclude_domains=_split_csv(exclude_domains),
            include_answer=True,
        )
    except TavilyError as exc:
        return {
            "tool": "tavily_search",
            "summary": f"TAVILY_SEARCH_UNAVAILABLE: {exc}",
            "data": [],
            "limitations": ["Tavily 暂不可用，未完成联网搜索。"],
        }

    items = _build_search_items(result.get("results") or [], max_results)
    summary = result.get("answer") or (f"共检索到 {len(items)} 条网络结果。" if items else "未检索到相关网络结果。")
    return {
        "tool": "tavily_search",
        "summary": summary,
        "data": items,
        "sources": [item.get("url") for item in items if item.get("url") and item.get("url") != "无链接"],
    }


async def tavily_research(
    query: str,
    include_domains: str = "",
    exclude_domains: str = "",
    max_results: int = 5,
) -> dict:
    try:
        result = await _tavily_research(
            query=query,
            max_results=max_results,
            include_domains=_split_csv(include_domains),
            exclude_domains=_split_csv(exclude_domains),
        )
    except TavilyError as exc:
        return {
            "tool": "tavily_research",
            "summary": f"TAVILY_RESEARCH_UNAVAILABLE: {exc}",
            "data": [],
            "limitations": ["Tavily 暂不可用，未完成深度研究。"],
        }

    items = result.get("results") or result.get("sources") or []
    sources = [
        {
            "title": item.get("title") or "无标题",
            "url": item.get("url") or "无链接",
        }
        for item in items[:max_results]
    ]
    summary = ""
    for key in ("summary", "answer", "report"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            summary = value.strip()
            break
    if not summary:
        summary = "未返回可用研究总结。"
    return {
        "tool": "tavily_research",
        "summary": summary,
        "data": sources,
        "sources": [item["url"] for item in sources if item["url"] != "无链接"],
    }


async def tavily_extract(urls: str) -> dict:
    url_list = [item.strip() for item in urls.replace("\n", ",").split(",") if item.strip()]
    if not url_list:
        return {
            "tool": "tavily_extract",
            "summary": "未提供可提取的 URL。",
            "data": [],
            "limitations": ["输入为空。"],
        }

    try:
        result = await _tavily_extract(url_list)
    except TavilyError as exc:
        return {
            "tool": "tavily_extract",
            "summary": f"TAVILY_EXTRACT_UNAVAILABLE: {exc}",
            "data": [],
            "limitations": ["Tavily 暂不可用，未完成页面抽取。"],
        }

    items = result.get("results") or result.get("data") or []
    data = []
    for idx, item in enumerate(items[:5], start=1):
        url = item.get("url") or url_list[min(idx - 1, len(url_list) - 1)]
        raw = (item.get("raw_content") or item.get("content") or "").strip()
        data.append(
            {
                "url": url,
                "content": raw[:800] + ("..." if len(raw) > 800 else ""),
            }
        )
    summary = f"已提取 {len(data)} 个页面内容。" if data else "未提取到页面内容。"
    return {
        "tool": "tavily_extract",
        "summary": summary,
        "data": data,
        "sources": [item["url"] for item in data],
    }


async def batch_tavily_search(
    queries: str,
    max_results_per_query: int = 3,
    time_range: str = "",
) -> dict:
    query_list = [q.strip() for q in queries.split("|||") if q.strip()]
    if not query_list:
        return {
            "tool": "batch_tavily_search",
            "summary": "未提供搜索关键词。",
            "data": [],
            "limitations": ["输入为空。"],
        }

    async def _single_search(query: str) -> dict[str, Any]:
        try:
            result = await _tavily_search(
                query=query,
                max_results=max_results_per_query,
                time_range=time_range or None,
                include_answer=True,
            )
        except TavilyError as exc:
            return {
                "query": query,
                "summary": f"搜索失败：{exc}",
                "results": [],
            }

        return {
            "query": query,
            "summary": (result.get("answer") or "").strip() or f"共检索到 {len(result.get('results') or [])} 条结果。",
            "results": _build_search_items(result.get("results") or [], max_results_per_query),
        }

    data = await asyncio.gather(*[_single_search(query) for query in query_list])
    return {
        "tool": "batch_tavily_search",
        "summary": f"已完成 {len(data)} 个关键词的联网搜索。",
        "data": data,
        "sources": [
            item["url"]
            for group in data
            for item in group.get("results", [])
            if item.get("url") and item.get("url") != "无链接"
        ],
    }
