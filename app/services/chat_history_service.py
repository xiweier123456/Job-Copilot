"""
app/services/chat_history_service.py

保存会话 transcript 的持久层。
这里存的是完整对话事实，而不是向量索引；
即使后续新增 Milvus 语义召回，完整历史仍然以 transcript 为准。
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HISTORY_ROOT = PROJECT_ROOT / "outputs" / "chat_history"

_HISTORY_LOCK = asyncio.Lock()


def _safe_session_id(session_id: str) -> str:
    """清洗 session_id，避免把非法字符直接带进文件名。"""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (session_id or "default").strip())
    return cleaned or "default"


def _history_file(session_id: str) -> Path:
    """把 session_id 映射到 transcript 文件路径。"""
    return HISTORY_ROOT / f"{_safe_session_id(session_id)}.json"


def _read_history_sync(session_id: str) -> list[dict[str, Any]]:
    """
    从磁盘读取某个 session 的 transcript。
    这是同步文件 IO，所以外层会用 asyncio.to_thread 包起来。
    """
    path = _history_file(session_id)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    return data if isinstance(data, list) else []


def _write_history_sync(session_id: str, turns: list[dict[str, Any]]) -> None:
    """把完整 turns 列表落盘为当前 session 的 transcript 文件。"""
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
    _history_file(session_id).write_text(
        json.dumps(turns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def get_chat_history(session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    """
    异步读取某个 session 的 transcript。
    limit 只截取最近几轮，适合给恢复上下文或历史接口使用。
    """
    turns = await asyncio.to_thread(_read_history_sync, session_id)
    if limit is not None and limit > 0:
        return turns[-limit:]
    return turns


async def save_chat_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """
    追加保存一轮对话到 transcript。
    用锁串行化写入，避免并发请求把同一个 session 的文件写乱。
    """
    session_id = str(turn.get("session_id") or "default")

    async with _HISTORY_LOCK:
        turns = await asyncio.to_thread(_read_history_sync, session_id)
        turns.append(turn)
        await asyncio.to_thread(_write_history_sync, session_id, turns)

    return turn


async def clear_chat_history(session_id: str) -> None:
    """
    删除指定 session 的 transcript。
    这里删除的是完整事实历史，清空后 /chat/history 不应再把旧消息恢复回来。
    """
    path = _history_file(session_id)

    async with _HISTORY_LOCK:
        if path.exists():
            await asyncio.to_thread(path.unlink)


async def build_history_context(session_id: str, limit: int = 4) -> str:
    """
    把最近几轮 transcript 拼成一段恢复提示词。
    它只该在 thread memory 不可用时兜底注入，避免和运行时记忆重复。
    """
    turns = await get_chat_history(session_id, limit=limit)
    if not turns:
        return ""

    lines = [
        "以下是这个 session 最近几轮已经持久化保存的对话记录。",
        "它用于在服务重启或页面重新进入后补回上下文，因此请把它当作历史事实参考，而不是当前用户的新问题。",
    ]

    for index, turn in enumerate(turns, start=1):
        user_message = str(turn.get("user_message") or "").strip()
        assistant_message = str(turn.get("assistant_message") or "").strip()
        status = str(turn.get("status") or "done").strip()
        created_at = str(turn.get("created_at") or "")

        if not user_message and not assistant_message:
            continue

        lines.append(f"[历史第 {index} 轮 | status={status} | created_at={created_at}]")
        if user_message:
            lines.append(f"用户：{user_message}")
        if assistant_message:
            lines.append(f"助手：{assistant_message}")
    lines.append("以上是历史对话记录的结束。")
    return "\n".join(lines).strip()
