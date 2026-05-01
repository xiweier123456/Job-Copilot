"""
app/agents/graph.py
使用 deepagents.create_deep_agent 构建 Job Copilot 主 agent。

设计说明：
- 主 agent 负责理解用户意图、决定是否委派任务、汇总最终回复
- subagents 负责执行特定求职任务：岗位检索、简历匹配、职业路径、面试准备
- 所有底层能力复用现有 MCP tools / services / RAG，不重造数据层
- skills 用于沉淀稳定行为规范，减少 system prompt 里堆过多规则
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from threading import Lock
from typing import Any, AsyncIterator

import httpx
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend
from deepagents.backends.filesystem import FilesystemBackend
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from app.config import settings
from app.agents.model_registry import AgentModelSpec, resolve_agent_model_spec
from app.prompts.agent_prompts import (
    build_career_agent_prompt,
    build_interview_agent_prompt,
    build_job_search_agent_prompt,
    build_main_system_prompt,
    build_resume_agent_prompt,
)
from app.prompts.service_prompts import build_context_compression_messages
from app.services import cache_service
from app.services.llm_client import chat_json_completion
from app.services.chat_history_service import build_history_context
from app.rag.chat_memory_store import search_chat_memory
from app.mcp.tool_registry import (
    get_subagent_tools,
    get_tool_spec_by_agent_name,
    serialize_tool_spec,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMORY_ROOT = PROJECT_ROOT / "outputs" / "memories"


SUBAGENT_NAMES = {
    "job-search-agent",
    "resume-agent",
    "career-agent",
    "interview-agent",
}
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
LATIN_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")
TOOL_NAME_PATTERN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*_tool)\b")
TRACE_SUBAGENT_PATTERN = re.compile(r"subagent_type['\"]:\s*['\"]([^'\"]+)['\"]")
TRACE_TOOL_PATTERN = re.compile(r"tool_calls=\[\{['\"]name['\"]:\s*['\"]([^'\"]+)['\"]")
TRACE_CONTEXT_TOOL_PATTERN = re.compile(r'"tool_call"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"')

class AgentRunError(RuntimeError):
    """当 deep agent 运行失败时抛出的统一异常。"""


class SessionBusyError(RuntimeError):
    """当同一个 session 已经有运行中的 agent 时抛出。"""

    def __init__(self, session_id: str, run_id: str):
        self.session_id = session_id
        self.run_id = run_id
        super().__init__(f"session {session_id!r} is already running: {run_id}")


@dataclass
class ActiveRun:
    """记录一个正在运行的 agent 实例，并暴露取消信号。"""

    run_id: str
    session_id: str
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class ToolLedgerEntry:
    """记录一次真实工具调用的生命周期。"""

    event_id: str
    name: str
    args_hash: str
    status: str = "started"
    started_at: float = field(default_factory=perf_counter)
    ended_at: float | None = None
    latency_ms: float | None = None
    error: str | None = None


class RunLedger:
    """一轮 agent 运行的结构化流水账，用于替代事后正则猜测工具调用。"""

    def __init__(self) -> None:
        self.tool_entries: list[ToolLedgerEntry] = []
        self._active_tools: dict[str, ToolLedgerEntry] = {}

    @staticmethod
    def _event_id(event: dict[str, Any], name: str) -> str:
        value = event.get("run_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        return f"{name}:{id(event)}"

    @staticmethod
    def _args_hash(event: dict[str, Any]) -> str:
        data = event.get("data") or {}
        tool_input = data.get("input") if isinstance(data, dict) else None
        if tool_input is None:
            return ""
        encoded = json.dumps(tool_input, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @staticmethod
    def _error_text(event: dict[str, Any]) -> str:
        data = event.get("data") or {}
        if isinstance(data, dict):
            for key in ("error", "exception"):
                value = data.get(key)
                if value:
                    return str(value)
        return ""

    def record_tool_start(self, event: dict[str, Any]) -> ToolLedgerEntry | None:
        name = event.get("name")
        if not isinstance(name, str) or not name or name == "task":
            return None

        entry = ToolLedgerEntry(
            event_id=self._event_id(event, name),
            name=name,
            args_hash=self._args_hash(event),
        )
        self.tool_entries.append(entry)
        self._active_tools[entry.event_id] = entry
        return entry

    def _finish_tool(self, event: dict[str, Any], status: str) -> ToolLedgerEntry | None:
        name = event.get("name")
        if not isinstance(name, str) or not name or name == "task":
            return None

        event_id = self._event_id(event, name)
        entry = self._active_tools.pop(event_id, None)
        if entry is None:
            for active_id, active_entry in reversed(list(self._active_tools.items())):
                if active_entry.name == name:
                    entry = self._active_tools.pop(active_id)
                    break
        if entry is None:
            entry = ToolLedgerEntry(
                event_id=event_id,
                name=name,
                args_hash=self._args_hash(event),
            )
            self.tool_entries.append(entry)

        entry.status = status
        entry.ended_at = perf_counter()
        entry.latency_ms = round((entry.ended_at - entry.started_at) * 1000, 2)
        if status == "error":
            entry.error = self._error_text(event) or "tool call failed"
        return entry

    def record_tool_end(self, event: dict[str, Any]) -> ToolLedgerEntry | None:
        return self._finish_tool(event, "completed")

    def record_tool_error(self, event: dict[str, Any]) -> ToolLedgerEntry | None:
        return self._finish_tool(event, "error")

    def tool_names(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for entry in self.tool_entries:
            if entry.name not in seen:
                seen.add(entry.name)
                names.append(entry.name)
        return names

    def tool_details(self) -> list[dict[str, Any]]:
        return [_build_tool_call_detail(entry) for entry in self.tool_entries]


_ACTIVE_RUNS: dict[str, ActiveRun] = {}
_ACTIVE_SESSION_RUNS: dict[str, str] = {}
_RUN_REGISTRY_LOCK = Lock()
_CHECKPOINTER: Any | None = None
_CHECKPOINTER_CONTEXT: Any | None = None
_CHECKPOINTER_LOCK = asyncio.Lock()
_AGENT_CACHE: dict[str, Any] = {}
_AGENT_LOCK = asyncio.Lock()


def _normalize_session_id(session_id: str) -> str:
    """把空 session 统一映射到 default，避免并发注册绕过。"""
    normalized = (session_id or "default").strip()
    return normalized or "default"


def _active_run_key(session_id: str) -> str:
    return cache_service.build_cache_key("run", "active", session_id)


def _run_status_key(run_id: str) -> str:
    return cache_service.build_cache_key("run", "status", run_id)


def create_run(session_id: str) -> ActiveRun:
    """为指定 session 创建并登记一个新的运行实例。"""
    normalized_session_id = _normalize_session_id(session_id)
    with _RUN_REGISTRY_LOCK:
        existing_run_id = _ACTIVE_SESSION_RUNS.get(normalized_session_id)
        if existing_run_id:
            raise SessionBusyError(normalized_session_id, existing_run_id)

        run = ActiveRun(run_id=uuid.uuid4().hex, session_id=normalized_session_id)
        redis_lock = cache_service.acquire_lock_sync(
            _active_run_key(normalized_session_id),
            run.run_id,
            settings.redis_run_lock_ttl_seconds,
        )
        if redis_lock is False:
            redis_run_id = cache_service.get_text_sync(_active_run_key(normalized_session_id)) or "unknown"
            raise SessionBusyError(normalized_session_id, redis_run_id)

        _ACTIVE_RUNS[run.run_id] = run
        _ACTIVE_SESSION_RUNS[normalized_session_id] = run.run_id

    cache_service.set_text_sync(
        _run_status_key(run.run_id),
        "running",
        ttl_seconds=settings.redis_run_lock_ttl_seconds,
    )
    return run


def get_run(run_id: str) -> ActiveRun | None:
    """根据 run_id 读取仍在跟踪中的运行实例。"""
    with _RUN_REGISTRY_LOCK:
        return _ACTIVE_RUNS.get(run_id)


def cancel_run(run_id: str) -> bool:
    """标记某个运行实例为已取消，并返回它是否存在。"""
    run = get_run(run_id)
    existing_status = cache_service.get_text_sync(_run_status_key(run_id))
    cache_service.set_text_sync(_run_status_key(run_id), "cancelled", ttl_seconds=settings.redis_run_lock_ttl_seconds)
    if run is not None:
        run.cancel_event.set()
        return True
    return existing_status is not None


def remove_run(run_id: str) -> None:
    """从内存中的活动运行表里移除一个运行实例。"""
    with _RUN_REGISTRY_LOCK:
        run = _ACTIVE_RUNS.pop(run_id, None)
        if run and _ACTIVE_SESSION_RUNS.get(run.session_id) == run_id:
            _ACTIVE_SESSION_RUNS.pop(run.session_id, None)
    if run:
        cache_service.release_lock_sync(_active_run_key(run.session_id), run_id)
    if cache_service.get_text_sync(_run_status_key(run_id)) != "cancelled":
        cache_service.set_text_sync(_run_status_key(run_id), "done", ttl_seconds=settings.redis_default_ttl_seconds)


def _is_run_cancelled(run_id: str) -> bool:
    run = get_run(run_id)
    if run and run.cancel_event.is_set():
        return True
    return cache_service.get_text_sync(_run_status_key(run_id)) == "cancelled"


async def has_active_thread_memory(session_id: str) -> bool:
    """检查这个 session 是否已有可继续复用的 thread checkpoint。"""
    checkpointer = await get_checkpointer()
    checkpoint = await checkpointer.aget_tuple({"configurable": {"thread_id": session_id}})
    return checkpoint is not None


async def clear_session_runtime_state(session_id: str) -> None:
    """
    清掉某个 session 的运行时线程状态。
    这里删的是 checkpointer 里的 thread checkpoint，并顺手取消该 session 下正在跑的 run。
    """
    checkpointer = await get_checkpointer()
    normalized_session_id = _normalize_session_id(session_id)
    await _delete_checkpoint_thread(checkpointer, normalized_session_id)

    with _RUN_REGISTRY_LOCK:
        runs = list(_ACTIVE_RUNS.values())

    for run in runs:
        if run.session_id == normalized_session_id:
            run.cancel_event.set()



SYSTEM_PROMPT = build_main_system_prompt(MEMORY_ROOT)


def _build_model(model_spec: AgentModelSpec):
    """构造 deep agent 使用的聊天模型。"""
    return init_chat_model(
        model=f"openai:{model_spec.model}",
        api_key=model_spec.api_key,
        base_url=model_spec.base_url,
        temperature=0.3,
    )


def _resolve_checkpoint_sqlite_path() -> Path:
    """解析 SQLite checkpoint 文件路径，并确保父目录存在。"""
    path = Path(settings.checkpoint_sqlite_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def _build_sqlite_checkpointer() -> tuple[Any, Any | None]:
    """创建 SQLite checkpointer；依赖缺失时回退到 MemorySaver。"""
    try:
        sqlite_checkpoint_module = importlib.import_module("langgraph.checkpoint.sqlite.aio")
        AsyncSqliteSaver = getattr(sqlite_checkpoint_module, "AsyncSqliteSaver")
    except ImportError:
        logger.warning(
            "CHECKPOINT_BACKEND=sqlite but langgraph-checkpoint-sqlite is not installed; "
            "falling back to in-memory checkpointing."
        )
        return MemorySaver(), None

    path = _resolve_checkpoint_sqlite_path()
    context = AsyncSqliteSaver.from_conn_string(str(path))
    if hasattr(context, "__aenter__"):
        checkpointer = await context.__aenter__()
        logger.info("Using SQLite LangGraph checkpoint store: %s", path)
        return checkpointer, context

    logger.info("Using SQLite LangGraph checkpoint store: %s", path)
    return context, None


async def get_checkpointer():
    """返回共享的 LangGraph checkpointer，用于保存线程级记忆。"""
    global _CHECKPOINTER, _CHECKPOINTER_CONTEXT

    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    async with _CHECKPOINTER_LOCK:
        if _CHECKPOINTER is not None:
            return _CHECKPOINTER

        if settings.checkpoint_backend == "sqlite":
            _CHECKPOINTER, _CHECKPOINTER_CONTEXT = await _build_sqlite_checkpointer()
        else:
            logger.info("Using in-memory LangGraph checkpoint store.")
            _CHECKPOINTER = MemorySaver()
            _CHECKPOINTER_CONTEXT = None
        return _CHECKPOINTER


async def _delete_checkpoint_thread(checkpointer: Any, thread_id: str) -> None:
    """兼容不同 checkpointer 实现的 thread 删除接口。"""
    if hasattr(checkpointer, "adelete_thread"):
        await checkpointer.adelete_thread(thread_id)
        return
    if hasattr(checkpointer, "delete_thread"):
        await asyncio.to_thread(checkpointer.delete_thread, thread_id)
        return
    logger.warning("Configured checkpointer does not support deleting thread %s", thread_id)


async def close_agent_runtime() -> None:
    """关闭 agent 运行时资源，主要用于释放 SQLite checkpoint 连接。"""
    global _CHECKPOINTER, _CHECKPOINTER_CONTEXT

    async with _AGENT_LOCK:
        _AGENT_CACHE.clear()

    async with _CHECKPOINTER_LOCK:
        context = _CHECKPOINTER_CONTEXT
        _CHECKPOINTER = None
        _CHECKPOINTER_CONTEXT = None

    if context is not None and hasattr(context, "__aexit__"):
        await context.__aexit__(None, None, None)

    await cache_service.close_cache()


def get_store():
    """返回 deep agent 运行时共用的内存存储。"""
    return InMemoryStore()


async def get_agent(model_provider: str | None = None):
    """创建单例 deep agent，并给每个 subagent 绑定对应工具集。"""
    model_spec = resolve_agent_model_spec(model_provider)
    cache_key = model_spec.provider

    if cache_key in _AGENT_CACHE:
        return _AGENT_CACHE[cache_key]

    async with _AGENT_LOCK:
        if cache_key in _AGENT_CACHE:
            return _AGENT_CACHE[cache_key]

        agent = await _build_agent(model_spec)
        _AGENT_CACHE[cache_key] = agent
        return agent


async def _build_agent(model_spec: AgentModelSpec):
    """创建 deep agent 实例。"""
    model = _build_model(model_spec)
    checkpointer = await get_checkpointer()
    store = get_store()

    def _make_backend(runtime):
        """把仓库文件访问路由到文件系统，把记忆目录路由到 StoreBackend。"""
        return CompositeBackend(
            default=FilesystemBackend(root_dir=str(PROJECT_ROOT),virtual_mode=True),
            routes={
                str(MEMORY_ROOT): StoreBackend(runtime),
            },

        )

    # ═══════════════════════════════════════════════════════════════
    # Subagent 定义
    # ═══════════════════════════════════════════════════════════════
    # Prompt 设计准则：
    #   1. 结构化分层：Role → Scope → Rules → Workflow (CoT) → Output → Few-shot
    #   2. XML 标签分隔各区域，帮助 LLM 精准定位
    #   3. Few-shot 示例展示期望的输入→输出模式
    #   4. Chain-of-Thought 在 workflow 中引导"分析→检索→综合→输出"
    #   5. Negative examples 说明不希望看到的回答
    #   6. 公共 Tavily 指令已收敛到 app.prompts.shared / agent_prompts，避免重复
    subagents = [
        {
            "name": "job-search-agent",
            "description": (
                "用于岗位检索、岗位对比、岗位要求总结、市场需求分析，以及基于真实招聘信息回答\"有哪些岗位\""
                "\"某岗位通常要求什么\"、\"某城市有哪些相关岗位\"等问题。"
            ),
            "system_prompt": build_job_search_agent_prompt(),
            "tools": get_subagent_tools("job-search-agent"),
            "skills": [
                "skills/tavily/tavily-search",
                "skills/tavily/tavily-research",
            ],
        },
        {
            "name": "resume-agent",
            "description": (
                "用于简历与目标岗位的匹配分析、识别优势与缺口、发现证据不足之处，并给出可执行的简历修改建议。"
                "适用于用户贴出简历、询问\"我的简历适合什么岗位\"或\"和某岗位匹不匹配\"等场景。"
            ),
            "system_prompt": build_resume_agent_prompt(),
            "tools": get_subagent_tools("resume-agent"),
            "skills": [
                "skills/tavily/tavily-search",
                "skills/tavily/tavily-research",
            ],
        },
        {
            "name": "career-agent",
            "description": (
                "用于根据用户的学历、专业、研究方向、项目经历、技能栈和目标城市推荐职业方向，"
                "并给出可执行的求职准备路径。适用于\"我适合投什么岗位\"、\"研究生该走什么方向\"、"
                "\"下一步怎么准备\"等问题。"
            ),
            "system_prompt": build_career_agent_prompt(),
            "tools": get_subagent_tools("career-agent"),
            "skills": [
                "skills/tavily/tavily-search",
                "skills/tavily/tavily-research",
            ],
        },
        {
            "name": "interview-agent",
            "description": (
                "用于面试准备：生成高频面试题、整理回答思路、提炼准备重点，并输出面试前检查清单。"
                "适用于\"这个岗位面试怎么准备\"、\"常见面试题有哪些\"、\"面经重点是什么\"等问题。"
            ),
            "system_prompt": build_interview_agent_prompt(),
            "tools": get_subagent_tools("interview-agent"),
            "skills": [
                "skills/tavily/tavily-search",
                "skills/tavily/tavily-research",
            ],
        },
    ]

    return create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[],
        subagents=subagents,
        skills=[
            "skills/tavily/tavily-search",
            "skills/tavily/tavily-research",
        ],
        name="job-copilot-agent",
        checkpointer=checkpointer,
        store=store,
        backend=_make_backend,
    )


def _extract_text_from_message_content(content) -> str:
    """从 deep agent 返回的 message.content 中尽量提取可展示文本。"""
    if not content:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue

            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
                    continue

                for key in ("content", "value"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                        break

        return "\n".join(parts).strip()

    return ""


def _extract_message_role(message: Any) -> str:
    """从 LangChain 风格的消息对象中提取角色字段。"""
    role = getattr(message, "type", None) or getattr(message, "role", None)
    return role if isinstance(role, str) else ""


def _extract_message_name(message: Any) -> str:
    """提取消息上的 name 字段，用来识别 subagent 或工具名。"""
    name = getattr(message, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()

    additional_kwargs = getattr(message, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        value = additional_kwargs.get("name")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _collect_sources(text: str) -> list[str]:
    """从回复文本中提取以“链接：”开头的唯一来源地址。"""
    sources: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("链接："):
            url = line.removeprefix("链接：").strip()
            if url and url not in sources:
                sources.append(url)
    return sources


def _collect_used_subagents(messages: list[Any], event_texts: list[str] | None = None) -> list[str]:
    """从消息元信息和事件轨迹中推断本轮实际使用过的 subagent。"""
    used: list[str] = []
    for message in messages:
        name = _extract_message_name(message)
        if name in SUBAGENT_NAMES and name not in used:
            used.append(name)

    for text in event_texts or []:
        for subagent_name in SUBAGENT_NAMES:
            if subagent_name in text and subagent_name not in used:
                used.append(subagent_name)
        for match in TRACE_SUBAGENT_PATTERN.findall(text):
            if match in SUBAGENT_NAMES and match not in used:
                used.append(match)

    return used


def _collect_tool_calls_summary(messages: list[Any], event_texts: list[str] | None = None) -> list[str]:
    """汇总消息与事件轨迹里出现过的工具 wrapper 名称。"""
    summaries: list[str] = []
    seen: set[str] = set()

    for message in messages:
        text = _extract_text_from_message_content(getattr(message, "content", None))
        if not text:
            continue

        for match in TOOL_NAME_PATTERN.findall(text):
            if match not in seen:
                seen.add(match)
                summaries.append(match)

    for text in event_texts or []:
        for match in TRACE_TOOL_PATTERN.findall(text):
            if match not in seen:
                seen.add(match)
                summaries.append(match)
        for match in TRACE_CONTEXT_TOOL_PATTERN.findall(text):
            if match not in seen:
                seen.add(match)
                summaries.append(match)

    return summaries


def _build_tool_call_details(tool_names: list[str]) -> list[dict[str, Any]]:
    """把工具名列表映射成适合前端展示的结构化工具信息。"""
    details: list[dict[str, Any]] = []
    for tool_name in tool_names:
        spec = get_tool_spec_by_agent_name(tool_name)
        if spec is None:
            details.append(
                {
                    "name": tool_name,
                    "agent_name": tool_name,
                    "display_name": tool_name,
                    "description": "",
                    "category": "unknown",
                    "requires_network": None,
                    "latency": None,
                    "evidence_type": None,
                    "status": "completed",
                }
            )
            continue
        details.append(serialize_tool_spec(spec, name=tool_name, status="completed"))
    return details


def _build_tool_call_detail(entry: ToolLedgerEntry) -> dict[str, Any]:
    """把一次真实工具调用记录映射成前端和历史可消费的结构。"""
    spec = get_tool_spec_by_agent_name(entry.name)
    if spec is None:
        payload = {
            "name": entry.name,
            "agent_name": entry.name,
            "display_name": entry.name,
            "description": "",
            "category": "unknown",
            "requires_network": None,
            "latency": None,
            "evidence_type": None,
            "status": entry.status,
        }
    else:
        payload = serialize_tool_spec(spec, name=entry.name, status=entry.status)

    payload.update(
        {
            "runtime_status": entry.status,
            "latency_ms": entry.latency_ms,
            "args_hash": entry.args_hash,
            "error": entry.error,
        }
    )
    return payload


def _build_run_trace(
    *,
    run_id: str,
    session_id: str,
    started_at: str,
    completed_at: str,
    latency_ms: float,
    used_subagents: list[str],
    tool_calls: list[dict[str, Any]],
    sources: list[str],
    event_count: int,
    context_compression: dict[str, Any],
) -> dict[str, Any]:
    """Build a frontend-friendly trace graph summary for a completed run."""
    failed_tools = [item for item in tool_calls if item.get("runtime_status") == "error"]
    network_tools = [item for item in tool_calls if item.get("requires_network") is True]
    timeline: list[dict[str, Any]] = [
        {
            "type": "run",
            "status": "started",
            "label": "Agent run started",
            "timestamp": started_at,
        }
    ]
    timeline.extend(
        {
            "type": "subagent",
            "status": "started",
            "label": name,
            "name": name,
        }
        for name in used_subagents
    )
    timeline.extend(
        {
            "type": "tool",
            "status": item.get("runtime_status") or item.get("status"),
            "label": item.get("display_name") or item.get("name"),
            "name": item.get("name"),
            "category": item.get("category"),
            "latency_ms": item.get("latency_ms"),
            "requires_network": item.get("requires_network"),
            "source_type": item.get("source_type"),
            "source_name": item.get("source_name"),
            "error": item.get("error"),
        }
        for item in tool_calls
    )
    timeline.append(
        {
            "type": "run",
            "status": "completed",
            "label": "Agent run completed",
            "timestamp": completed_at,
            "latency_ms": latency_ms,
        }
    )

    return {
        "trace_id": run_id,
        "run_id": run_id,
        "session_id": session_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": latency_ms,
        "event_count": event_count,
        "metrics": {
            "subagent_count": len(used_subagents),
            "tool_call_count": len(tool_calls),
            "network_tool_call_count": len(network_tools),
            "failed_tool_call_count": len(failed_tools),
            "source_count": len(sources),
            "context_compression_applied": bool(context_compression.get("applied")),
        },
        "subagents": [{"name": name, "status": "started"} for name in used_subagents],
        "tools": tool_calls,
        "sources": sources,
        "context_compression": context_compression,
        "timeline": timeline,
    }


def _extract_reply(result: dict[str, Any]) -> str:
    """从 deep agent 的结果结构中尽量提取最终回复文本。"""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        text = _extract_text_from_message_content(content)
        if text and _extract_message_role(msg) in {"ai", "assistant"}:
            return text

    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        text = _extract_text_from_message_content(content)
        if text:
            return text

    for key in ("output", "final_output", "response"):
        value = result.get(key)
        text = _extract_text_from_message_content(value)
        if text:
            return text
        if isinstance(value, str) and value.strip():
            return value.strip()

    return "抱歉，我暂时无法从 agent 结果中提取有效回复。"


def _utc_timestamp() -> str:
    """返回当前 UTC 时间的 ISO-8601 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


def _extract_todo_items(text: str) -> list[dict[str, str]]:
    """从原始事件文本中提取 TodoWrite 风格的任务项，用于 SSE 进度展示。"""
    items: list[dict[str, str]] = []

    for status in ("in_progress", "pending", "completed"):
        pattern = re.compile(
            rf"['\"]content['\"]:\s*['\"]([^'\"]+)['\"].*?['\"]status['\"]:\s*['\"]{status}['\"]"
        )
        for content in pattern.findall(text):
            item = {"content": content.strip(), "status": status}
            if item not in items:
                items.append(item)

    return items


def _build_stream_event(
    event_type: str,
    session_id: str,
    run_id: str,
    sequence: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """构造一条发给 API 层的标准化 SSE 事件。"""
    return {
        "type": event_type,
        "session_id": session_id,
        "run_id": run_id,
        "sequence": sequence,
        "timestamp": _utc_timestamp(),
        "payload": payload,
    }


def _build_memory_context(memories: list[dict[str, Any]]) -> str:
    """把长期记忆结果压缩成一个有限长度的提示块。"""
    if not memories:
        return ""

    lines = [
        "以下是与当前问题相关的长期记忆，仅作为补充上下文，不要覆盖当前用户输入。",
    ]
    total_chars = len(lines[0])

    for item in memories:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        created_at = str(item.get("created_at") or "")
        role_scope = str(item.get("role_scope") or "")
        prefix = f"- [{role_scope or 'memory'}"
        if created_at:
            prefix += f" | {created_at}"
        prefix += f"] {text}"

        if total_chars + len(prefix) > settings.memory_prompt_char_budget:
            break
        lines.append(prefix)
        total_chars += len(prefix)

    if len(lines) == 1:
        return ""
    return "\n".join(lines).strip()


def _estimate_token_count(text: str) -> int:
    """用轻量规则估算 token 数，避免为压缩判断引入额外 tokenizer 依赖。"""
    if not text:
        return 0

    cjk_count = len(CJK_PATTERN.findall(text))
    non_cjk_text = CJK_PATTERN.sub(" ", text)
    latin_token_count = 0
    for token in LATIN_TOKEN_PATTERN.findall(non_cjk_text):
        if token.isspace():
            continue
        if re.fullmatch(r"[A-Za-z0-9_]+", token):
            latin_token_count += max(1, round(len(token) / 4))
        else:
            latin_token_count += 1
    return cjk_count + latin_token_count


def _truncate_for_compression(text: str) -> str:
    """压缩输入过长时保留开头结构和最近信息，降低压缩调用本身超长的风险。"""
    max_chars = settings.context_compression_max_input_chars
    if len(text) <= max_chars:
        return text

    recent_chars = min(settings.context_compression_fallback_recent_chars, max_chars // 2)
    head_chars = max_chars - recent_chars
    return (
        text[:head_chars].rstrip()
        + "\n\n...[中间上下文过长，已在压缩前省略]...\n\n"
        + text[-recent_chars:].lstrip()
    )


def _fallback_compressed_context(supplemental_context: str) -> str:
    """LLM 压缩失败时的保守兜底：保留最近上下文。"""
    recent_chars = settings.context_compression_fallback_recent_chars
    text = supplemental_context.strip()
    if len(text) <= recent_chars:
        return text
    return "以下是因压缩服务不可用而保留的最近上下文片段：\n" + text[-recent_chars:].lstrip()


async def _compress_supplemental_context(
    *,
    history_context: str,
    memory_context: str,
    current_message: str,
) -> tuple[str, dict[str, Any]]:
    """在超出预算时压缩补充上下文，当前用户消息保持原文。"""
    supplemental_context = "\n\n".join(
        part.strip()
        for part in (
            f"【历史对话】\n{history_context}" if history_context else "",
            f"【长期记忆】\n{memory_context}" if memory_context else "",
        )
        if part.strip()
    )
    raw_runtime_message = _build_runtime_message(current_message, history_context, memory_context)
    original_tokens = _estimate_token_count(raw_runtime_message)
    meta: dict[str, Any] = {
        "enabled": settings.context_compression_enabled,
        "applied": False,
        "original_tokens_estimate": original_tokens,
        "final_tokens_estimate": original_tokens,
        "trigger_tokens": settings.context_compression_trigger_tokens,
        "target_tokens": settings.context_compression_target_tokens,
        "error": None,
    }

    should_skip = (
        not settings.context_compression_enabled
        or not supplemental_context
        or original_tokens <= settings.context_compression_trigger_tokens
    )
    if should_skip:
        return raw_runtime_message, meta

    compressed_context = ""
    try:
        parsed = await chat_json_completion(
            messages=build_context_compression_messages(
                current_message=current_message,
                supplemental_context=_truncate_for_compression(supplemental_context),
                target_tokens=settings.context_compression_target_tokens,
            ),
            temperature=0.1,
            max_tokens=min(1800, max(600, settings.context_compression_target_tokens)),
        )
        compressed_context = str(parsed.get("compressed_context") or "").strip()
    except Exception as exc:
        logger.warning("Failed to compress runtime context: %s", exc)
        meta["error"] = str(exc)

    if not compressed_context:
        compressed_context = _fallback_compressed_context(supplemental_context)
        meta["error"] = meta["error"] or "empty_compression_result"

    runtime_message = (
        "以下是压缩后的历史上下文和长期记忆，仅作为补充参考，不要覆盖当前用户问题：\n"
        f"{compressed_context.strip()}\n\n"
        f"{current_message.strip()}"
    ).strip()
    meta["applied"] = True
    meta["final_tokens_estimate"] = _estimate_token_count(runtime_message)
    return runtime_message, meta


def _build_runtime_message(message: str, history_context: str, memory_context: str) -> str:
    """在需要时把历史上下文和长期记忆拼进当前用户消息。"""
    parts = [part.strip() for part in (history_context, memory_context, message) if part and part.strip()]
    return "\n\n".join(parts).strip()


def _normalize_event(
    event: dict[str, Any],
    session_id: str,
    run_id: str,
    sequence: int,
    ledger: RunLedger,
    seen_subagents: set[str],
    seen_todos: set[str],
) -> dict[str, Any] | None:
    """把 deep agent 原始事件转换成前端可消费的稳定流式事件。"""
    event_name = event.get("event")
    name = event.get("name")
    event_text = json.dumps(event, ensure_ascii=False, default=str)

    todo_items = _extract_todo_items(event_text)
    if todo_items:
        todo_key = json.dumps(todo_items, ensure_ascii=False, sort_keys=True)
        if todo_key not in seen_todos:
            seen_todos.add(todo_key)
            return _build_stream_event(
                "todo",
                session_id,
                run_id,
                sequence,
                {"items": todo_items},
            )

    if event_name == "on_tool_start" and name == "task":
        return _build_stream_event(
            "status",
            session_id,
            run_id,
            sequence,
            {"stage": "thinking", "message": "正在规划并拆解任务"},
        )

    if event_name == "on_tool_start" and isinstance(name, str) and name != "task":
        entry = ledger.record_tool_start(event)
        payload = _build_tool_call_detail(entry) if entry is not None else {"name": name, "status": "started"}
        return _build_stream_event(
            "tool",
            session_id,
            run_id,
            sequence,
            payload,
        )

    if event_name == "on_tool_end" and isinstance(name, str) and name != "task":
        entry = ledger.record_tool_end(event)
        payload = _build_tool_call_detail(entry) if entry is not None else {"name": name, "status": "completed"}
        return _build_stream_event(
            "tool",
            session_id,
            run_id,
            sequence,
            payload,
        )

    if event_name == "on_tool_error" and isinstance(name, str) and name != "task":
        entry = ledger.record_tool_error(event)
        payload = _build_tool_call_detail(entry) if entry is not None else {"name": name, "status": "error"}
        return _build_stream_event(
            "tool",
            session_id,
            run_id,
            sequence,
            payload,
        )

    for subagent_name in SUBAGENT_NAMES:
        if subagent_name in event_text and subagent_name not in seen_subagents:
            seen_subagents.add(subagent_name)
            return _build_stream_event(
                "subagent",
                session_id,
                run_id,
                sequence,
                {"name": subagent_name, "status": "started"},
            )

    return None


def _build_final_response(
    result: dict[str, Any],
    session_id: str,
    run_id: str,
    model_spec: AgentModelSpec,
    started_at: str,
    event_texts: list[str],
    ledger: RunLedger,
    context_compression: dict[str, Any],
    start: float,
) -> dict[str, Any]:
    """根据原始 agent 结果和事件轨迹组装最终 API 返回结构。"""
    messages = result.get("messages", [])
    reply = _extract_reply(result)
    used_subagents = _collect_used_subagents(messages, event_texts)
    tool_calls_summary = ledger.tool_names()
    tool_calls = ledger.tool_details()
    sources = _collect_sources(reply)
    latency_ms = round((perf_counter() - start) * 1000, 2)
    completed_at = _utc_timestamp()
    trace = _build_run_trace(
        run_id=run_id,
        session_id=session_id,
        started_at=started_at,
        completed_at=completed_at,
        latency_ms=latency_ms,
        used_subagents=used_subagents,
        tool_calls=tool_calls,
        sources=sources,
        event_count=len(event_texts),
        context_compression=context_compression,
    )

    logger.info(
        "Agent run completed: session_id=%s, used_subagents=%s, tools=%s, latency_ms=%s",
        session_id,
        used_subagents,
        tool_calls_summary,
        latency_ms,
    )

    return {
        "reply": reply,
        "session_id": session_id,
        "run_id": run_id,
        "model_provider": model_spec.provider,
        "model_name": model_spec.model,
        "used_subagents": used_subagents,
        "tool_calls_summary": tool_calls_summary,
        "tool_calls": tool_calls,
        "sources": sources,
        "latency_ms": latency_ms,
        "context_compression": context_compression,
        "trace": trace,
        "error": None,
    }


async def stream_agent_events(
    message: str,
    session_id: str = "default",
    run_id: str | None = None,
    model_provider: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """流式产出标准化运行事件，供 API 层转发为 SSE。"""
    session_id = _normalize_session_id(session_id)
    if run_id is None:
        run = create_run(session_id)
        run_id = run.run_id

    model_spec = resolve_agent_model_spec(model_provider)
    agent = await get_agent(model_spec.provider)
    start = perf_counter()
    started_at = _utc_timestamp()
    final_result: dict[str, Any] | None = None
    event_texts: list[str] = []
    ledger = RunLedger()
    seen_subagents: set[str] = set()
    seen_todos: set[str] = set()
    sequence = 0
    current_run_id = run_id
    history_context = ""
    if not await has_active_thread_memory(session_id):
        history_context = await build_history_context(
            session_id,
            limit=settings.memory_recent_history_limit,
        )

    memory_context = ""
    memories = search_chat_memory(message, session_id, settings.memory_recall_top_k)
    if memories:
        memory_context = _build_memory_context(memories)

    yield _build_stream_event(
        "status",
        session_id,
        current_run_id,
        sequence,
        {"stage": "started", "message": "已开始处理请求"},
    )
    sequence += 1

    should_try_compression = (
        settings.context_compression_enabled
        and (history_context or memory_context)
        and _estimate_token_count(_build_runtime_message(message, history_context, memory_context))
        > settings.context_compression_trigger_tokens
    )
    if should_try_compression:
        yield _build_stream_event(
            "status",
            session_id,
            current_run_id,
            sequence,
            {"stage": "compressing", "message": "上下文较长，正在压缩历史信息"},
        )
        sequence += 1

    runtime_message, context_compression = await _compress_supplemental_context(
        history_context=history_context,
        memory_context=memory_context,
        current_message=message,
    )

    if context_compression.get("applied"):
        yield _build_stream_event(
            "status",
            session_id,
            current_run_id,
            sequence,
            {
                "stage": "compressed",
                "message": "已压缩历史上下文",
                "original_tokens_estimate": context_compression.get("original_tokens_estimate"),
                "final_tokens_estimate": context_compression.get("final_tokens_estimate"),
            },
        )
        sequence += 1

    try:
        async for event in agent.astream_events(
            {
                "messages": [{"role": "user", "content": runtime_message}],
                "metadata": {"session_id": session_id, "run_id": current_run_id},
            },
            config={"configurable": {"thread_id": session_id}, "recursion_limit": 100},
            version="v2",
        ):
            if _is_run_cancelled(current_run_id):
                yield _build_stream_event(
                    "stopped",
                    session_id,
                    current_run_id,
                    sequence,
                    {"reason": "user_cancelled", "message": "生成已停止"},
                )
                return

            event_text = str(event)
            event_texts.append(event_text)

            normalized = _normalize_event(event, session_id, current_run_id, sequence, ledger, seen_subagents, seen_todos)
            if normalized is not None:
                yield normalized
                sequence += 1

            if event.get("event") == "on_chain_end" and event.get("name") == "job-copilot-agent":
                data = event.get("data") or {}
                output = data.get("output")
                if isinstance(output, dict):
                    final_result = output
    except asyncio.CancelledError:
        yield _build_stream_event(
            "stopped",
            session_id,
            current_run_id,
            sequence,
            {"reason": "client_disconnected", "message": "生成已停止"},
        )
        return
    except httpx.RemoteProtocolError as exc:
        logger.warning("Upstream model stream closed early for session %s: %s", session_id, exc)
        yield _build_stream_event(
            "error",
            session_id,
            current_run_id,
            sequence,
            {"message": "上游模型流式连接中断，请稍后重试。"},
        )
        raise AgentRunError("上游模型流式连接中断，请稍后重试。") from exc
    except Exception as exc:
        logger.exception("Agent run failed for session %s", session_id)
        yield _build_stream_event(
            "error",
            session_id,
            current_run_id,
            sequence,
            {"message": "agent 运行失败，请稍后重试。"},
        )
        raise AgentRunError("agent 运行失败，请稍后重试。") from exc
    finally:
        remove_run(current_run_id)

    yield _build_stream_event(
        "status",
        session_id,
        current_run_id,
        sequence,
        {"stage": "finalizing", "message": "正在整理最终结果"},
    )
    sequence += 1

    yield _build_stream_event(
        "final",
        session_id,
        current_run_id,
        sequence,
        _build_final_response(
            final_result or {},
            session_id,
            current_run_id,
            model_spec,
            started_at,
            event_texts,
            ledger,
            context_compression,
            start,
        ),
    )


async def run_agent(message: str, session_id: str = "default", model_provider: str | None = None) -> dict[str, Any]:
    """运行主 agent，并返回可直接给 API 层使用的结构化结果。"""
    final_payload: dict[str, Any] | None = None

    async for stream_event in stream_agent_events(message, session_id, model_provider=model_provider):
        if stream_event["type"] == "final":
            final_payload = stream_event["payload"]

    if final_payload is None:
        raise AgentRunError("agent 运行失败，请稍后重试。")

    return final_payload
