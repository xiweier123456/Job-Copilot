"""
app/rag/chat_memory_store.py
基于 mem0 + Milvus 的对话长期记忆封装。

职责边界：
- 完整 transcript 仍然保存在 chat_history_service 中
- 这里负责长期语义记忆的写入、搜索与清空
- 对外继续保持 save/search/clear 接口，尽量不影响上层调用
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from mem0 import Memory

from app.config import settings

logger = logging.getLogger(__name__)

_GENERIC_MEMORY_PATTERNS = (
    "适合什么岗位",
    "适合哪些岗位",
    "推荐岗位",
    "什么工作",
    "帮我看看",
    "分析一下",
    "怎么找工作",
    "我想问问",
)

_HIGH_SIGNAL_USER_MARKERS = (
    "我会",
    "我有",
    "我擅长",
    "我希望",
    "我想找",
    "目标城市",
    "目标岗位",
    "学历",
    "专业",
    "经验",
    "实习",
    "项目",
    "技能",
    "简历",
    "薪资",
    "城市",
)


def _resolve_history_db_path() -> str:
    path = Path(settings.mem0_history_db_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _build_mem0_config() -> dict[str, Any]:
    embedder_model = settings.mem0_embedder_model or settings.embedding_model
    embedder_dims = settings.mem0_embedder_dims or settings.embedding_dim
    llm_model = settings.mem0_llm_model or settings.service_llm_model

    vector_config: dict[str, Any] = {
        "collection_name": settings.mem0_collection_name,
        "embedding_model_dims": embedder_dims,
        "url": f"http://{settings.milvus_host}:{settings.milvus_port}",
        "token": "",
        "db_name": "",
    }

    config: dict[str, Any] = {
        "vector_store": {
            "provider": "milvus",
            "config": vector_config,
        },
        "history_db_path": _resolve_history_db_path(),
        "version": "v1.1",
        "llm": {
            "provider": settings.mem0_llm_provider,
            "config": {
                "model": llm_model,
                "api_key": settings.service_llm_api_key,
                "deepseek_base_url": settings.service_llm_base_url,
                "temperature": 0.1,
            },
        },
        "embedder": {
            "provider": settings.mem0_embedder_provider,
            "config": {
                "model": embedder_model,
                "embedding_dims": embedder_dims,
            },
        },
    }

    if settings.mem0_embedder_provider == "openai":
        config["embedder"]["config"].update(
            {
                "api_key": settings.service_llm_api_key,
                "openai_base_url": settings.service_llm_base_url,
            }
        )

    if settings.mem0_embedder_provider == "huggingface":
        config["embedder"]["config"].pop("embedding_dims", None)

    return config


@lru_cache(maxsize=1)
def _get_memory_client() -> Memory:
    return Memory.from_config(_build_mem0_config())


def _session_memory_scope(session_id: str) -> dict[str, str]:
    safe_session_id = (session_id or "default").strip() or "default"
    return {
        "user_id": safe_session_id,
        "agent_id": "job-copilot-chat-memory",
    }


def _normalize_memory_result(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    return {
        "memory_id": item.get("id", ""),
        "session_id": item.get("user_id") or metadata.get("session_id", ""),
        "role_scope": metadata.get("role_scope", ""),
        "text": item.get("memory", ""),
        "created_at": item.get("created_at", ""),
        "status": metadata.get("status", ""),
        "score": round(float(item.get("score", 0.0)), 4) if item.get("score") is not None else 0.0,
    }


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _is_generic_user_message(user_message: str) -> bool:
    normalized = "".join(user_message.split())
    if not normalized:
        return True
    if len(normalized) < settings.memory_min_user_chars:
        return True
    return any(pattern in normalized for pattern in _GENERIC_MEMORY_PATTERNS)


def _is_high_signal_user_message(user_message: str) -> bool:
    normalized = "".join(user_message.split())
    if len(normalized) < settings.memory_min_user_chars:
        return False
    return any(marker in normalized for marker in _HIGH_SIGNAL_USER_MARKERS)


def _has_context_signal(context: dict[str, Any]) -> bool:
    return any(str(context.get(key) or "").strip() for key in ("target_city", "job_direction", "user_profile", "resume_text"))


def _should_save_turn_memory(turn: dict[str, Any]) -> bool:
    if str(turn.get("status") or "done").strip() != "done":
        return False

    user_message = str(turn.get("user_message") or "").strip()
    assistant_message = str(turn.get("assistant_message") or "").strip()
    context = turn.get("context") or {}

    if not assistant_message:
        return False
    if _has_context_signal(context):
        return True
    if len(assistant_message) < settings.memory_min_assistant_chars:
        return False
    return _is_high_signal_user_message(user_message) and not _is_generic_user_message(user_message)


def _build_memory_text(turn: dict[str, Any]) -> str:
    context = turn.get("context") or {}
    lines: list[str] = ["Career memory from a completed Job Copilot turn."]

    context_fields = (
        ("target_city", "Target city"),
        ("job_direction", "Job direction"),
        ("user_profile", "User profile"),
        ("resume_text", "Resume facts"),
    )
    for key, label in context_fields:
        value = _clip_text(context.get(key), settings.memory_context_summary_chars)
        if value:
            lines.append(f"- {label}: {value}")

    user_message = _clip_text(turn.get("user_message"), 500)
    assistant_message = _clip_text(turn.get("assistant_message"), settings.memory_context_summary_chars)
    if user_message and not _is_generic_user_message(user_message):
        lines.append(f"- User goal or preference: {user_message}")
    if assistant_message:
        lines.append(f"- Assistant recommendation: {assistant_message}")

    return "\n".join(lines).strip()


async def save_chat_memory(turn: dict) -> None:
    """
    把一轮对话写入长期语义记忆。
    user 消息单独存；assistant 只有成功完成时才和 user 组成整轮文本入库。
    """
    if not settings.memory_enabled:
        return

    user_message = str(turn.get("user_message") or "").strip()
    assistant_message = str(turn.get("assistant_message") or "").strip()
    status = str(turn.get("status") or "done").strip()
    session_id = str(turn.get("session_id") or "default")
    created_at = str(turn.get("created_at") or "")
    turn_id = str(turn.get("turn_id") or "")
    scope = _session_memory_scope(session_id)
    client = _get_memory_client()

    try:
        if settings.memory_save_user_raw and _is_high_signal_user_message(user_message):
            client.add(
                [{"role": "user", "content": user_message}],
                metadata={
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "created_at": created_at,
                    "role_scope": "user",
                    "status": status,
                },
                infer=False,
                **scope,
            )

        if _should_save_turn_memory(turn):
            memory_text = _build_memory_text(turn)
            client.add(
                [{"role": "user", "content": memory_text}],
                metadata={
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "created_at": created_at,
                    "role_scope": "career_context",
                    "status": status,
                    "memory_kind": "completed_turn_summary",
                },
                infer=True,
                **scope,
            )
    except Exception:
        logger.exception("Failed to save chat memory for session %s", session_id)


async def clear_chat_memory(session_id: str) -> None:
    """清空某个 session 的长期语义记忆。"""
    if not settings.memory_enabled:
        return

    client = _get_memory_client()
    try:
        client.delete_all(**_session_memory_scope(session_id))
    except Exception:
        logger.exception("Failed to clear chat memory for session %s", session_id)


def search_chat_memory(query: str, session_id: str, top_k: int = 3) -> list[dict]:
    """按语义相似度召回某个 session 下最相关的历史记忆。"""
    if not settings.memory_enabled or not query.strip():
        return []

    client = _get_memory_client()
    try:
        response = client.search(
            query=query,
            limit=top_k,
            threshold=settings.memory_recall_score_threshold,
            **_session_memory_scope(session_id),
        )
    except Exception:
        logger.exception("Failed to search chat memory for session %s", session_id)
        return []

    results = response.get("results") or []
    return [_normalize_memory_result(item) for item in results]
