from __future__ import annotations

import json

import httpx
import pytest
import respx


@pytest.mark.asyncio
@respx.mock
async def test_tavily_client_posts_expected_payload(monkeypatch):
    """The low-level Tavily client should send the expected httpx request."""
    from app.services import tavily_client

    monkeypatch.setattr(tavily_client.settings, "tavily_api_key", "test-key")
    route = respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "找到 1 条结果",
                "results": [{"title": "示例", "url": "https://example.com", "content": "正文"}],
            },
        )
    )

    result = await tavily_client.tavily_search(
        "数据分析师 面经",
        max_results=1,
        include_answer=True,
        include_domains=["example.com"],
    )

    assert result["answer"] == "找到 1 条结果"
    assert route.called
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer test-key"
    payload = json.loads(request.content.decode("utf-8"))
    assert payload["query"] == "数据分析师 面经"
    assert payload["max_results"] == 1
    assert payload["include_answer"] is True
    assert payload["include_domains"] == ["example.com"]


@pytest.mark.asyncio
@respx.mock
async def test_tavily_tool_returns_stable_fallback_on_http_error(monkeypatch):
    """The MCP-facing Tavily tool should downgrade errors instead of raising."""
    from app.mcp.tools import tavily as tavily_tools
    from app.services import tavily_client

    monkeypatch.setattr(tavily_client.settings, "tavily_api_key", "test-key")
    monkeypatch.setattr(tavily_client.settings, "tool_allowed_domains", "")
    respx.post("https://api.tavily.com/search").mock(
        return_value=httpx.Response(500, text="upstream exploded")
    )

    result = await tavily_tools.tavily_search("算法工程师 面经", max_results=2)

    assert result["tool"] == "tavily_search"
    assert result["data"] == []
    assert result["limitations"] == ["Tavily 暂不可用，未完成联网搜索。"]
    assert result["summary"].startswith("TAVILY_SEARCH_UNAVAILABLE")


@pytest.mark.asyncio
async def test_tavily_extract_blocks_urls_outside_allowlist(monkeypatch):
    """The MCP-facing extract tool should enforce the configured domain allowlist."""
    from app.mcp.tools import tavily as tavily_tools
    from app.services import tavily_client

    monkeypatch.setattr(tavily_client.settings, "tool_security_enabled", True)
    monkeypatch.setattr(tavily_client.settings, "tool_allowed_domains", "example.com")

    result = await tavily_tools.tavily_extract("https://evil.test/private")

    assert result["tool"] == "tavily_extract"
    assert result["data"] == []
    assert result["summary"].startswith("TOOL_SECURITY_BLOCKED")
    assert result["security"]["domain_policy"] == "allowlist"
    assert result["security"]["blocked"] == ["https://evil.test/private"]
