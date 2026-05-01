from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services import chat_history_service


@pytest.mark.asyncio
async def test_list_chat_sessions_returns_recent_summaries(monkeypatch):
    history_root = Path("outputs/test_chat_history_service")
    history_root.mkdir(parents=True, exist_ok=True)
    for path in history_root.glob("*.json"):
        path.unlink()
    monkeypatch.setattr(chat_history_service, "HISTORY_ROOT", history_root)

    try:
        (history_root / "session-a.json").write_text(
            json.dumps(
                [
                    {
                        "session_id": "session-a",
                        "user_message": "我想找北京的数据分析岗位",
                        "assistant_message": "可以优先看数据分析师。",
                        "created_at": "2026-05-01T08:00:00+00:00",
                    },
                    {
                        "session_id": "session-a",
                        "user_message": "简历怎么改",
                        "assistant_message": "突出 SQL 和项目指标。",
                        "created_at": "2026-05-01T09:00:00+00:00",
                    },
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (history_root / "session-b.json").write_text(
            json.dumps(
                [
                    {
                        "session_id": "session-b",
                        "user_message": "面试题怎么准备",
                        "assistant_message": "",
                        "created_at": "2026-05-01T07:00:00+00:00",
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        sessions = await chat_history_service.list_chat_sessions()

        assert [item["session_id"] for item in sessions] == ["session-a", "session-b"]
        assert sessions[0]["title"] == "我想找北京的数据分析岗位"
        assert sessions[0]["turn_count"] == 2
        assert sessions[0]["message_count"] == 4
        assert sessions[0]["updated_at"] == "2026-05-01T09:00:00+00:00"
    finally:
        for path in history_root.glob("*.json"):
            path.unlink()
        history_root.rmdir()
