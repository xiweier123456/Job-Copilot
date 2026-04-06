"""
统一维护 MCP 与 deep agent 共用的工具注册表。

这个模块的目标是把“逻辑工具”的定义集中到一处，避免 MCP 注册、
agent wrapper 导出、subagent 工具分组在多个文件里逐渐漂移。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from langchain_core.tools import tool

from app.mcp.tools.search_jobs import search_jobs
from app.mcp.tools.tavily import (
    batch_tavily_search,
    tavily_extract,
    tavily_research,
    tavily_search,
)

ToolCategory = Literal["job_db", "web_search", "web_extract"]
LatencyLevel = Literal["low", "medium", "high"]
EvidenceType = Literal[
    "job_postings",
    "web_results",
    "research_summary",
    "page_content",
    "batched_web_results",
]


@dataclass(frozen=True)
class ToolSpec:
    """描述一个逻辑工具在 MCP 层和 agent 层的统一元信息。"""

    key: str
    agent_name: str
    display_name: str
    description: str
    category: ToolCategory
    requires_network: bool
    latency: LatencyLevel
    evidence_type: EvidenceType
    enabled: bool
    raw_callable: Callable
    agent_callable: Callable


@tool
async def search_jobs_tool(query: str, city: str = "", industry: str = "", top_k: int = 5) -> dict:
    """检索岗位库中的代表性岗位证据。

    适用场景：
    - 用户想看某个方向有哪些真实岗位
    - 用户想看某个城市或行业的岗位样本
    - 用户想基于真实 JD 总结常见要求

    参数：
    - query：岗位 / 技能 / 求职方向关键词
    - city：可选，城市过滤
    - industry：可选，行业过滤
    - top_k：返回岗位数量

    返回结构化岗位证据。应把它当作回答素材，而不是最终回复。
    """
    return await search_jobs(
        query=query,
        city=city or None,
        industry=industry or None,
        top_k=top_k,
    )


@tool
async def tavily_search_tool(
    query: str,
    include_domains: str = "",
    exclude_domains: str = "",
    max_results: int = 5,
    topic: str = "general",
    time_range: str = "",
) -> dict:
    """执行 Tavily 联网搜索，补充最新网页证据。

    适用场景：
    - 用户需要近期网络信息
    - 需要搜文章、公司页面、面经、社区讨论
    - 在回答岗位 / 简历 / 面试问题前补充联网依据

    典型例子：
    - 帮我搜一下 2026 算法工程师校招面经
    - 找一些牛客和知乎上的真实经验
    - 看看这个岗位最近的真实要求

    返回 MCP 层统一格式的搜索结果。
    """
    return await tavily_search(
        query=query,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        max_results=max_results,
        topic=topic,
        time_range=time_range,
    )


@tool
async def tavily_research_tool(
    query: str,
    include_domains: str = "",
    exclude_domains: str = "",
    max_results: int = 5,
) -> dict:
    """执行更深层的 Tavily 研究模式，用于多来源综合分析。"""
    return await tavily_research(
        query=query,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        max_results=max_results,
    )


@tool
async def tavily_extract_tool(urls: str) -> dict:
    """抽取一个或多个网页的正文内容，供后续分析使用。"""
    return await tavily_extract(urls=urls)


@tool
async def batch_tavily_search_tool(
    queries: str,
    max_results_per_query: int = 3,
    time_range: str = "",
) -> dict:
    """并发执行多个 Tavily 搜索，适合多关键词批量检索。"""
    return await batch_tavily_search(
        queries=queries,
        max_results_per_query=max_results_per_query,
        time_range=time_range,
    )


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        key="search_jobs",
        agent_name="search_jobs_tool",
        display_name="岗位检索",
        description="从岗位数据库中检索代表性岗位样本，用于给出真实招聘证据。",
        category="job_db",
        requires_network=False,
        latency="medium",
        evidence_type="job_postings",
        enabled=True,
        raw_callable=search_jobs,
        agent_callable=search_jobs_tool,
    ),
    ToolSpec(
        key="tavily_search",
        agent_name="tavily_search_tool",
        display_name="联网搜索",
        description="执行快速联网搜索，返回最新网页结果与摘要。",
        category="web_search",
        requires_network=True,
        latency="medium",
        evidence_type="web_results",
        enabled=True,
        raw_callable=tavily_search,
        agent_callable=tavily_search_tool,
    ),
    ToolSpec(
        key="tavily_research",
        agent_name="tavily_research_tool",
        display_name="深度研究",
        description="执行多来源联网研究，适合趋势、比较和总结类问题。",
        category="web_search",
        requires_network=True,
        latency="high",
        evidence_type="research_summary",
        enabled=True,
        raw_callable=tavily_research,
        agent_callable=tavily_research_tool,
    ),
    ToolSpec(
        key="tavily_extract",
        agent_name="tavily_extract_tool",
        display_name="网页抽取",
        description="根据 URL 抽取网页正文内容，供后续解析和引用。",
        category="web_extract",
        requires_network=True,
        latency="medium",
        evidence_type="page_content",
        enabled=True,
        raw_callable=tavily_extract,
        agent_callable=tavily_extract_tool,
    ),
    ToolSpec(
        key="batch_tavily_search",
        agent_name="batch_tavily_search_tool",
        display_name="批量联网搜索",
        description="并发搜索多个关键词，适合批量收集求职相关网络证据。",
        category="web_search",
        requires_network=True,
        latency="high",
        evidence_type="batched_web_results",
        enabled=True,
        raw_callable=batch_tavily_search,
        agent_callable=batch_tavily_search_tool,
    ),
)

_TOOL_SPEC_BY_KEY = {spec.key: spec for spec in TOOL_SPECS}
_TOOL_SPEC_BY_AGENT_NAME = {spec.agent_name: spec for spec in TOOL_SPECS}

SUBAGENT_TOOL_KEYS: dict[str, tuple[str, ...]] = {
    "job-search-agent": (
        "search_jobs",
        "tavily_search",
        "tavily_research",
        "tavily_extract",
        "batch_tavily_search",
    ),
    "resume-agent": (
        "search_jobs",
        "tavily_search",
        "tavily_research",
        "tavily_extract",
        "batch_tavily_search",
    ),
    "career-agent": (
        "search_jobs",
        "tavily_search",
        "tavily_research",
        "batch_tavily_search",
    ),
    "interview-agent": (
        "search_jobs",
        "tavily_search",
        "tavily_research",
        "tavily_extract",
        "batch_tavily_search",
    ),
}


def get_tool_specs() -> tuple[ToolSpec, ...]:
    """返回当前启用的全部工具规格。"""
    return tuple(spec for spec in TOOL_SPECS if spec.enabled)


def get_tool_spec(tool_key: str) -> ToolSpec:
    """按逻辑工具 key 读取单个工具规格。"""
    return _TOOL_SPEC_BY_KEY[tool_key]


def get_tool_spec_by_agent_name(agent_name: str) -> ToolSpec | None:
    """按 agent wrapper 名称读取工具规格；不存在时返回 None。"""
    return _TOOL_SPEC_BY_AGENT_NAME.get(agent_name)


def serialize_tool_spec(spec: ToolSpec, *, name: str | None = None, status: str | None = None) -> dict:
    """把工具规格整理成适合 API / 前端消费的结构化字典。"""
    return {
        "name": name or spec.agent_name,
        "agent_name": spec.agent_name,
        "display_name": spec.display_name,
        "description": spec.description,
        "category": spec.category,
        "requires_network": spec.requires_network,
        "latency": spec.latency,
        "evidence_type": spec.evidence_type,
        "status": status,
    }


def get_mcp_tools() -> list[Callable]:
    """返回应该暴露给 FastMCP 的原始工具函数。"""
    return [spec.raw_callable for spec in get_tool_specs()]


def get_all_agent_tools() -> list[Callable]:
    """按稳定顺序返回全部 agent wrapper。"""
    return [spec.agent_callable for spec in get_tool_specs()]


def get_subagent_tools(subagent_name: str) -> list[Callable]:
    """返回指定 subagent 可用的 agent 工具集合。"""
    tool_keys = SUBAGENT_TOOL_KEYS[subagent_name]
    return [
        _TOOL_SPEC_BY_KEY[key].agent_callable
        for key in tool_keys
        if _TOOL_SPEC_BY_KEY[key].enabled
    ]


def get_subagent_tool_specs(subagent_name: str) -> list[ToolSpec]:
    """返回指定 subagent 的工具元信息，便于后续做展示或调度决策。"""
    tool_keys = SUBAGENT_TOOL_KEYS[subagent_name]
    return [
        _TOOL_SPEC_BY_KEY[key]
        for key in tool_keys
        if _TOOL_SPEC_BY_KEY[key].enabled
    ]
