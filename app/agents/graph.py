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
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend
from deepagents.backends.filesystem import FilesystemBackend
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from app.config import settings
from app.prompts.agent_prompts import (
    build_career_agent_prompt,
    build_interview_agent_prompt,
    build_job_search_agent_prompt,
    build_main_system_prompt,
    build_resume_agent_prompt,
)
from app.services.chat_history_service import build_history_context
from app.agents.tools import (
    search_jobs_tool,
    tavily_search_tool,
    tavily_research_tool,
    tavily_extract_tool,
    batch_tavily_search_tool,
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
TOOL_NAME_PATTERN = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*_tool)\b")
TRACE_SUBAGENT_PATTERN = re.compile(r"subagent_type['\"]:\s*['\"]([^'\"]+)['\"]")
TRACE_TOOL_PATTERN = re.compile(r"tool_calls=\[\{['\"]name['\"]:\s*['\"]([^'\"]+)['\"]")
TRACE_CONTEXT_TOOL_PATTERN = re.compile(r'"tool_call"\s*:\s*\{\s*"name"\s*:\s*"([^"]+)"')


class AgentRunError(RuntimeError):
    """Raised when the deep agent run fails."""


@dataclass
class ActiveRun:#表示一个正在运行的 agent 实例，包含 run_id、session_id 和一个 asyncio.Event 用于取消运行
    run_id: str
    session_id: str
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


_ACTIVE_RUNS: dict[str, ActiveRun] = {}#全局字典，存储所有正在运行的 agent 实例，key 是 run_id，value 是 ActiveRun 对象


def create_run(session_id: str) -> ActiveRun:#创建一个新的 agent 运行实例，生成唯一 run_id，并将其与 session_id 关联，存储在全局 _ACTIVE_RUNS 中
    run = ActiveRun(run_id=uuid.uuid4().hex, session_id=session_id)
    _ACTIVE_RUNS[run.run_id] = run
    return run

def get_run(run_id: str) -> ActiveRun | None:#根据 run_id 获取当前运行的 agent 实例，查看是否正在运行
    return _ACTIVE_RUNS.get(run_id)


def cancel_run(run_id: str) -> bool:#将当前运行的 agent 标记为取消，返回是否成功找到并标记
    run = get_run(run_id)
    if run is None:
        return False
    run.cancel_event.set()
    return True


def remove_run(run_id: str) -> None:#从全局 _ACTIVE_RUNS 中移除一个运行实例，通常在运行结束后调用
    _ACTIVE_RUNS.pop(run_id, None)


async def has_active_thread_memory(session_id: str) -> bool:
    """检查这个 session 在 MemorySaver 里是否已有可继续复用的 thread checkpoint。"""
    checkpointer = get_checkpointer()
    checkpoint = await checkpointer.aget_tuple({"configurable": {"thread_id": session_id}})
    return checkpoint is not None


async def clear_session_runtime_state(session_id: str) -> None:
    """
    清掉某个 session 的运行时线程状态。
    这里删的是 checkpointer 里的 thread checkpoint，并顺手取消该 session 下正在跑的 run。
    """
    checkpointer = get_checkpointer()
    await checkpointer.adelete_thread(session_id)

    for run in list(_ACTIVE_RUNS.values()):
        if run.session_id == session_id:
            run.cancel_event.set()



SYSTEM_PROMPT = build_main_system_prompt(MEMORY_ROOT)


def _build_model():
    """构造 deep agent 使用的聊天模型。"""
    return init_chat_model(
        model=f"openai:{settings.agent_llm_model}",
        api_key=settings.agent_llm_api_key,
        base_url=settings.agent_llm_base_url,
        temperature=0.3,
    )


@lru_cache(maxsize=1)
def get_checkpointer():
    return MemorySaver()


@lru_cache(maxsize=1)
def get_store():
    return InMemoryStore()


@lru_cache(maxsize=1)
def get_agent():
    """单例创建 deep agent，避免每次请求重复初始化。"""
    model = _build_model()
    checkpointer = get_checkpointer()
    store = get_store()

    def _make_backend(runtime):
        return CompositeBackend(
            default=FilesystemBackend(root_dir=str(PROJECT_ROOT)),
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
            "tools": [search_jobs_tool, tavily_search_tool, tavily_research_tool, tavily_extract_tool, batch_tavily_search_tool],
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
            "tools": [search_jobs_tool, tavily_search_tool, tavily_research_tool, tavily_extract_tool, batch_tavily_search_tool],
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
            "tools": [search_jobs_tool, tavily_search_tool, tavily_research_tool, batch_tavily_search_tool],
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
            "tools": [search_jobs_tool, tavily_search_tool, tavily_research_tool, tavily_extract_tool, batch_tavily_search_tool],
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
        skills=["skills/tavily"],
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
    role = getattr(message, "type", None) or getattr(message, "role", None)
    return role if isinstance(role, str) else ""


def _extract_message_name(message: Any) -> str:
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
    sources: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("链接："):
            url = line.removeprefix("链接：").strip()
            if url and url not in sources:
                sources.append(url)
    return sources


def _collect_used_subagents(messages: list[Any], event_texts: list[str] | None = None) -> list[str]:
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


def _extract_reply(result: dict[str, Any]) -> str:
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
    return datetime.now(timezone.utc).isoformat()


def _extract_todo_items(text: str) -> list[dict[str, str]]:
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
    return {
        "type": event_type,
        "session_id": session_id,
        "run_id": run_id,
        "sequence": sequence,
        "timestamp": _utc_timestamp(),
        "payload": payload,
    }


def _build_runtime_message(message: str, history_context: str) -> str:
    # thread_id=session_id 负责正常连续对话的主路径。
    # transcript 只在“当前进程内没有可用线程记忆”时作为恢复兜底，避免每轮都把同一份历史重复注入。
    if not history_context:
        return message
    return f"{history_context}\n\n{message}".strip()


def _normalize_event(
    event: dict[str, Any],
    session_id: str,
    run_id: str,
    sequence: int,
    seen_subagents: set[str],
    seen_todos: set[str],
) -> dict[str, Any] | None:
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
        return _build_stream_event(
            "tool",
            session_id,
            run_id,
            sequence,
            {"name": name, "status": "started"},
        )

    if event_name == "on_tool_end" and isinstance(name, str) and name != "task":
        return _build_stream_event(
            "tool",
            session_id,
            run_id,
            sequence,
            {"name": name, "status": "completed"},
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


def _build_final_response(result: dict[str, Any], session_id: str, event_texts: list[str], start: float) -> dict[str, Any]:
    messages = result.get("messages", [])
    reply = _extract_reply(result)
    used_subagents = _collect_used_subagents(messages, event_texts)
    tool_calls_summary = _collect_tool_calls_summary(messages, event_texts)
    sources = _collect_sources(reply)
    latency_ms = round((perf_counter() - start) * 1000, 2)

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
        "used_subagents": used_subagents,
        "tool_calls_summary": tool_calls_summary,
        "sources": sources,
        "latency_ms": latency_ms,
        "error": None,
    }


async def stream_agent_events(
    message: str,
    session_id: str = "default",
    run_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    agent = get_agent()
    start = perf_counter()
    final_result: dict[str, Any] | None = None
    event_texts: list[str] = []
    seen_subagents: set[str] = set()
    seen_todos: set[str] = set()
    sequence = 0
    current_run_id = run_id or uuid.uuid4().hex
    history_context = ""
    if not await has_active_thread_memory(session_id):
        history_context = await build_history_context(session_id)
    runtime_message = _build_runtime_message(message, history_context)

    yield _build_stream_event(
        "status",
        session_id,
        current_run_id,
        sequence,
        {"stage": "started", "message": "已开始处理请求"},
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
            run = get_run(current_run_id)
            if run and run.cancel_event.is_set():
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

            normalized = _normalize_event(event, session_id, current_run_id, sequence, seen_subagents, seen_todos)
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
        _build_final_response(final_result or {}, session_id, event_texts, start),
    )


async def run_agent(message: str, session_id: str = "default") -> dict[str, Any]:
    """运行主 agent，并返回可直接给 API 层使用的结构化结果。"""
    final_payload: dict[str, Any] | None = None

    async for stream_event in stream_agent_events(message, session_id):
        if stream_event["type"] == "final":
            final_payload = stream_event["payload"]

    if final_payload is None:
        raise AgentRunError("agent 运行失败，请稍后重试。")

    return final_payload
