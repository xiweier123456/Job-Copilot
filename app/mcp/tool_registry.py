"""
统一维护 MCP 与 deep agent 共用的工具注册表。

这个模块的目标是把“逻辑工具”的定义集中到一处，避免 MCP 注册、
agent wrapper 导出、subagent 工具分组在多个文件里逐渐漂移。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from threading import Lock, Thread
from time import monotonic
from typing import Any, Callable, Literal

from langchain_core.tools import StructuredTool, tool
from pydantic import Field, create_model

from app.config import ExternalMCPServiceConfig, settings
from app.mcp.external_service_registry import (
    discover_external_services,
    call_external_tool,
)
from app.mcp.tools.search_jobs import search_jobs
from app.mcp.tools.tavily import (
    batch_tavily_search,
    tavily_extract,
    tavily_research,
    tavily_search,
)

ToolCategory = Literal["job_db", "web_search", "web_extract", "external_mcp"]
LatencyLevel = Literal["low", "medium", "high"]
EvidenceType = Literal[
    "job_postings",
    "web_results",
    "research_summary",
    "page_content",
    "batched_web_results",
    "external_mcp",
]
SourceType = Literal["local", "external_mcp"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    """描述一个逻辑工具在 MCP 层和 agent 层的统一元信息。
        args:
        agent_name: 给 agent wrapper 使用的工具名称，必须符合 Python 标识符规范，且在本项目里保持唯一。
        display_name: 给人看的工具名称，可以包含空格和特殊字符。
        description: 工具功能的简要描述，供 MCP 使用者和 agent 开发者参考。
        category: 工具分类，便于做工具分组和调度决策。
        requires_network: 这个工具在执行时是否需要联网，便于做网络异常处理和调度决策。
        latency: 这个工具的预期响应速度，便于做调度决策。
        evidence_type: 这个工具返回的证据类型，便于做后续分析和决策。
        enabled: 这个工具当前是否启用，便于做灰度发布和功能开关。
        raw_callable: 这个工具的原始调用函数，直接暴露给 FastMCP。
        agent_callable: 这个工具给 agent 使用的包装函数，通常是一个 StructuredTool 实例。
        source_type: 这个工具的来源类型，区分本地定义和外部 MCP 服务。
        source_name: 这个工具的来源名称，对于本地工具通常是 "local"，对于外部工具是 MCP 服务的名称。
        canonical_name: 这个工具的稳定唯一标识，格式为 "{exposed_name}:{priority}"，用于解决外部工具冲突。
        raw_name: 这个工具在 MCP 层的原始名称，通常是 agent_name，但对于外部工具可能包含服务前缀。
        exposed_name: 这个工具在 MCP 层暴露的名称，通常是 agent_name，但对于外部工具可能经过调整以避免冲突。
    """

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
    source_type: SourceType = "local"
    source_name: str = "local"
    canonical_name: str | None = None
    raw_name: str | None = None
    exposed_name: str | None = None


@tool
async def search_jobs_tool(query: str, city: str = "", industry: str = "", top_k: int = 5) -> dict:
    """检索岗位库中的代表性岗位证据。"""
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
    """执行 Tavily 联网搜索，补充最新网页证据。"""
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


LOCAL_TOOL_SPECS: tuple[ToolSpec, ...] = (
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
        canonical_name="search_jobs",
        raw_name="search_jobs",
        exposed_name="search_jobs",
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
        canonical_name="tavily_search",
        raw_name="tavily_search",
        exposed_name="tavily_search",
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
        canonical_name="tavily_research",
        raw_name="tavily_research",
        exposed_name="tavily_research",
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
        canonical_name="tavily_extract",
        raw_name="tavily_extract",
        exposed_name="tavily_extract",
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
        canonical_name="batch_tavily_search",
        raw_name="batch_tavily_search",
        exposed_name="batch_tavily_search",
    ),
)

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


def _run_async(coro):
    """在同步上下文里安全执行异步发现逻辑。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - 这里只做跨线程转发
            error["value"] = exc

    thread = Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result.get("value", ())


def _normalize_external_description(description: str, service_name: str, tool_name: str = "") -> str:
    """给外部工具补齐并清洗可展示描述。"""
    haystack = f"{tool_name} {description}".lower()
    if any(keyword in haystack for keyword in ("understand_image", "image", "vision")):
        return "图片理解工具，支持本地路径或 URL 图片分析。"
    if any(keyword in haystack for keyword in ("web_search", "search", "research", "query")):
        return "实时网络搜索工具，返回搜索结果和相关建议。"
    if any(keyword in haystack for keyword in ("extract", "crawl", "fetch", "scrape")):
        return "信息提取工具，可从网页或资源中抽取结构化内容。"

    text = (description or "").strip()
    if not text:
        return f"来自外部 MCP 服务 {service_name} 的工具。"

    for marker in ("Args:", "Returns:", "Parameters:", "Search Strategy:"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("You MUST") or stripped.startswith("IMPORTANT:"):
            continue
        lines.append(stripped)

    text = " ".join(lines)
    if len(text) > 80:
        text = text[:77].rstrip() + "..."
    return text or f"来自外部 MCP 服务 {service_name} 的工具。"


def _infer_external_category(tool_name: str, description: str) -> ToolCategory:
    """根据名称和描述给外部工具一个保守分类。"""
    haystack = f"{tool_name} {description}".lower()
    if any(keyword in haystack for keyword in ("search", "research", "query")):
        return "web_search"
    if any(keyword in haystack for keyword in ("extract", "crawl", "fetch", "scrape")):
        return "web_extract"

    return "external_mcp"


def _build_external_raw_callable(service: ExternalMCPServiceConfig, source_name: str):
    """生成给 FastMCP 暴露的外部工具调用包装。"""
    async def _external_tool(**kwargs):
        return await call_external_tool(service, source_name, kwargs)

    _external_tool.__name__ = source_name
    _external_tool.__doc__ = f"Proxy to external MCP tool {source_name} from {service.name}."
    return _external_tool


def _build_args_schema(exposed_name: str, input_schema: dict[str, Any]):
    """把 MCP inputSchema 尽量转成一个宽松的 Pydantic 模型。"""
    properties = input_schema.get("properties") or {}
    required_fields = set(input_schema.get("required") or [])
    field_definitions: dict[str, tuple[Any, Any]] = {}

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    for field_name, spec in properties.items():
        if not isinstance(spec, dict):
            annotation = Any
            default = ... if field_name in required_fields else None
            field_definitions[field_name] = (annotation, default)
            continue

        json_type = spec.get("type")
        annotation = type_map.get(json_type, Any)
        description = spec.get("description") or None
        if field_name in required_fields:
            default = Field(..., description=description)
        else:
            default = Field(default=spec.get("default", None), description=description)
        field_definitions[field_name] = (annotation, default)

    if not field_definitions:
        return None

    return create_model(f"{exposed_name.title().replace('_', '')}Args", **field_definitions)


def _build_external_agent_callable(
    service: ExternalMCPServiceConfig,
    source_name: str,
    exposed_name: str,
    description: str,
    input_schema: dict[str, Any],
):
    """生成给 agent 使用的外部工具包装。"""
    async def _external_tool(**kwargs):
        return await call_external_tool(service, source_name, kwargs)

    args_schema = _build_args_schema(exposed_name, input_schema)
    return StructuredTool.from_function(
        coroutine=_external_tool,
        name=exposed_name,
        description=description,
        args_schema=args_schema,
        infer_schema=args_schema is None,
    )


def _resolve_external_collisions(specs: list[ToolSpec]) -> list[ToolSpec]:
    """对外部工具做稳定去重；本地工具永远优先。"""
    local_names = {spec.agent_name for spec in LOCAL_TOOL_SPECS}
    winners: dict[str, ToolSpec] = {}
    for spec in sorted(specs, key=lambda item: (item.source_name, -len(item.agent_name))):
        if spec.agent_name in local_names:
            continue
        current = winners.get(spec.agent_name)
        if current is None:
            winners[spec.agent_name] = spec
            continue
        current_priority = int(current.canonical_name.rsplit(":", 1)[-1]) if current.canonical_name else 0
        incoming_priority = int(spec.canonical_name.rsplit(":", 1)[-1]) if spec.canonical_name else 0
        if incoming_priority > current_priority:
            winners[spec.agent_name] = spec
    return list(winners.values())


def _build_external_tool_specs() -> tuple[ToolSpec, ...]:
    """发现外部 MCP 工具并转换为统一 ToolSpec。"""
    snapshots = _run_async(discover_external_services())
    collected: list[ToolSpec] = []
    for snapshot in snapshots:
        service = snapshot.config
        for item in snapshot.tools:
            description = _normalize_external_description(item.description, service.name, item.source_name)
            display_name = item.exposed_name
            raw_callable = _build_external_raw_callable(service, item.source_name)
            agent_callable = _build_external_agent_callable(
                service,
                item.source_name,
                item.exposed_name,
                description,
                item.input_schema,
            )
            collected.append(
                ToolSpec(
                    key=f"external::{service.name}::{item.source_name}",
                    agent_name=item.exposed_name,
                    display_name=display_name,
                    description=description,
                    category=_infer_external_category(item.source_name, description),
                    requires_network=item.requires_network,
                    latency="medium",
                    evidence_type="external_mcp",
                    enabled=True,
                    raw_callable=raw_callable,
                    agent_callable=agent_callable,
                    source_type="external_mcp",
                    source_name=service.name,
                    canonical_name=f"{item.exposed_name}:{item.priority}",
                    raw_name=item.source_name,
                    exposed_name=item.exposed_name,
                )
            )
    winners = _resolve_external_collisions(collected)
    return tuple(sorted(winners, key=lambda spec: (spec.source_name, spec.agent_name)))


_EXTERNAL_TOOL_CACHE_LOCK = Lock()
_EXTERNAL_TOOL_CACHE: tuple[ToolSpec, ...] | None = None
_EXTERNAL_TOOL_CACHE_EXPIRES_AT = 0.0
_EXTERNAL_TOOL_CACHE_RETRY_AT = 0.0


def _external_discovery_ttl_seconds() -> int:
    """读取外部 MCP 工具清单缓存时间，配置异常时使用保守默认值。"""
    value = getattr(settings, "mcp_external_discovery_ttl_seconds", 60)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 60


def _external_discovery_failure_ttl_seconds() -> int:
    """读取外部 MCP 发现失败后的重试冷却时间。"""
    value = getattr(settings, "mcp_external_discovery_failure_ttl_seconds", 30)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 30


def refresh_external_tool_specs() -> tuple[ToolSpec, ...]:
    """强制刷新外部 MCP 工具缓存，适合管理接口或调试脚本调用。"""
    return _get_cached_external_tool_specs(force_refresh=True)


def _get_cached_external_tool_specs(*, force_refresh: bool = False) -> tuple[ToolSpec, ...]:
    """返回带 TTL 缓存的外部 MCP 工具规格，避免运行时频繁连接远端服务。"""
    global _EXTERNAL_TOOL_CACHE, _EXTERNAL_TOOL_CACHE_EXPIRES_AT, _EXTERNAL_TOOL_CACHE_RETRY_AT

    now = monotonic()
    with _EXTERNAL_TOOL_CACHE_LOCK:
        cached = _EXTERNAL_TOOL_CACHE
        if not force_refresh and cached is not None and now < _EXTERNAL_TOOL_CACHE_EXPIRES_AT:
            return cached
        if not force_refresh and cached is not None and now < _EXTERNAL_TOOL_CACHE_RETRY_AT:
            return cached

    try:
        specs = _build_external_tool_specs()
    except Exception as exc:
        logger.warning("Failed to refresh external MCP tools: %s", exc)
        with _EXTERNAL_TOOL_CACHE_LOCK:
            cached = _EXTERNAL_TOOL_CACHE
            _EXTERNAL_TOOL_CACHE_RETRY_AT = monotonic() + _external_discovery_failure_ttl_seconds()
            if cached is not None:
                return cached
            _EXTERNAL_TOOL_CACHE = ()
        return ()

    with _EXTERNAL_TOOL_CACHE_LOCK:
        _EXTERNAL_TOOL_CACHE = specs
        _EXTERNAL_TOOL_CACHE_EXPIRES_AT = monotonic() + _external_discovery_ttl_seconds()
        _EXTERNAL_TOOL_CACHE_RETRY_AT = 0.0
    return specs


def get_tool_specs() -> tuple[ToolSpec, ...]:
    """返回当前启用的全部工具规格。"""
    local_specs = tuple(spec for spec in LOCAL_TOOL_SPECS if spec.enabled)
    external_specs = _get_cached_external_tool_specs()
    return local_specs + external_specs


def get_tool_spec(tool_key: str) -> ToolSpec:
    """按逻辑工具 key 读取单个工具规格。"""
    for spec in get_tool_specs():
        if spec.key == tool_key:
            return spec
    raise KeyError(tool_key)


def get_tool_spec_by_agent_name(agent_name: str) -> ToolSpec | None:
    """按 agent wrapper 名称读取工具规格；不存在时返回 None。"""
    for spec in get_tool_specs():
        if spec.agent_name == agent_name:
            return spec
    return None


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
        "source_type": spec.source_type,
        "source_name": spec.source_name,
        "canonical_name": spec.canonical_name,
        "raw_name": spec.raw_name,
        "exposed_name": spec.exposed_name,
    }


def get_mcp_tools() -> list[Callable]:
    """返回应该暴露给 FastMCP 的原始工具函数。

    目前仅暴露本地工具。外部 MCP 工具已经由其各自服务提供，
    这里不再二次注册到当前 FastMCP Server，避免动态代理函数
    因 ``**kwargs`` 签名无法通过 FastMCP 的工具注册校验。
    """
    return [spec.raw_callable for spec in get_tool_specs() if spec.source_type == "local"]


def get_all_agent_tools() -> list[Callable]:
    """按稳定顺序返回全部 agent wrapper。"""
    return [spec.agent_callable for spec in get_tool_specs()]


def get_subagent_tools(subagent_name: str) -> list[Callable]:
    """返回指定 subagent 可用的 agent 工具集合。"""
    allowed_local_keys = SUBAGENT_TOOL_KEYS[subagent_name]
    specs = get_tool_specs()
    local_specs = [spec for spec in specs if spec.key in allowed_local_keys and spec.enabled]
    external_specs = [spec for spec in specs if spec.source_type == "external_mcp" and spec.enabled]
    return [spec.agent_callable for spec in [*local_specs, *external_specs]]


def get_subagent_tool_specs(subagent_name: str) -> list[ToolSpec]:
    """返回指定 subagent 的工具元信息，便于后续做展示或调度决策。"""
    allowed_local_keys = SUBAGENT_TOOL_KEYS[subagent_name]
    specs = get_tool_specs()
    return [
        spec
        for spec in specs
        if spec.enabled and (spec.key in allowed_local_keys or spec.source_type == "external_mcp")
    ]
