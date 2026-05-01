from __future__ import annotations

import pytest

from app.rag import chat_memory_store


class FakeMemoryClient:
    def __init__(self):
        self.add_calls: list[dict] = []

    def add(self, messages, **kwargs):
        self.add_calls.append({"messages": messages, "kwargs": kwargs})


@pytest.mark.asyncio
async def test_save_chat_memory_skips_generic_question_without_context(monkeypatch):
    client = FakeMemoryClient()
    monkeypatch.setattr(chat_memory_store, "_get_memory_client", lambda: client)
    monkeypatch.setattr(chat_memory_store.settings, "memory_enabled", True)
    monkeypatch.setattr(chat_memory_store.settings, "memory_save_user_raw", False)
    monkeypatch.setattr(chat_memory_store.settings, "memory_min_assistant_chars", 20)

    await chat_memory_store.save_chat_memory(
        {
            "session_id": "s1",
            "turn_id": "t1",
            "status": "done",
            "created_at": "2026-05-01T00:00:00+00:00",
            "user_message": "我想问问我适合什么岗位",
            "assistant_message": "建议你先补充城市、简历、技能和项目经历，再做更可靠的岗位判断。",
            "context": {},
        }
    )

    assert client.add_calls == []


@pytest.mark.asyncio
async def test_save_chat_memory_writes_completed_context_summary_once(monkeypatch):
    client = FakeMemoryClient()
    monkeypatch.setattr(chat_memory_store, "_get_memory_client", lambda: client)
    monkeypatch.setattr(chat_memory_store.settings, "memory_enabled", True)
    monkeypatch.setattr(chat_memory_store.settings, "memory_save_user_raw", False)
    monkeypatch.setattr(chat_memory_store.settings, "memory_context_summary_chars", 200)

    await chat_memory_store.save_chat_memory(
        {
            "session_id": "s2",
            "turn_id": "t2",
            "status": "done",
            "created_at": "2026-05-01T00:00:00+00:00",
            "user_message": "我会 Python 和 SQL，希望找数据分析岗位",
            "assistant_message": "你更适合数据分析师、商业分析师和数据运营岗位，优先补强指标体系和 SQL 项目。",
            "context": {
                "target_city": "北京",
                "job_direction": "数据分析",
                "user_profile": "统计学硕士，会 Python、SQL，有 BI 看板项目。",
                "resume_text": "",
            },
        }
    )

    assert len(client.add_calls) == 1
    call = client.add_calls[0]
    assert call["kwargs"]["infer"] is True
    assert call["kwargs"]["metadata"]["role_scope"] == "career_context"
    assert call["kwargs"]["user_id"] == "s2"
    content = call["messages"][0]["content"]
    assert "Target city: 北京" in content
    assert "Job direction: 数据分析" in content
    assert "Assistant recommendation" in content
