from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import httpx
import pytest


@pytest.mark.asyncio
async def test_chat_stream_returns_sse_events_and_persists_final_turn(monkeypatch):
    """The chat endpoint should stream normalized SSE events and save the final turn."""
    saved_turns: list[dict] = []
    saved_memories: list[dict] = []

    graph_module = types.ModuleType("app.agents.graph")

    class AgentRunError(RuntimeError):
        pass

    class SessionBusyError(RuntimeError):
        def __init__(self, session_id: str, run_id: str):
            self.session_id = session_id
            self.run_id = run_id
            super().__init__(run_id)

    def create_run(session_id: str):
        assert session_id == "session-1"
        return SimpleNamespace(run_id="run-1")

    def cancel_run(run_id: str) -> bool:
        return run_id == "run-1"

    async def stream_agent_events(message: str, session_id: str, run_id: str, model_provider: str | None = None):
        assert session_id == "session-1"
        assert run_id == "run-1"
        assert model_provider is None
        assert "【用户背景】" in message
        assert "会 Python" in message
        yield {
            "type": "status",
            "session_id": session_id,
            "run_id": run_id,
            "sequence": 0,
            "timestamp": "2026-05-01T00:00:00+00:00",
            "payload": {"stage": "started", "message": "开始处理"},
        }
        yield {
            "type": "final",
            "session_id": session_id,
            "run_id": run_id,
            "sequence": 1,
            "timestamp": "2026-05-01T00:00:01+00:00",
            "payload": {
                "reply": "建议优先投递数据分析师。",
                "session_id": session_id,
                "used_subagents": ["career-agent"],
                "tool_calls_summary": ["search_jobs_tool"],
                "tool_calls": [],
                "sources": [],
                "latency_ms": 12.3,
                "context_compression": {"applied": False},
                "trace": {
                    "trace_id": "run-1",
                    "metrics": {"tool_call_count": 1},
                    "timeline": [{"type": "run", "status": "completed"}],
                },
                "error": None,
            },
        }

    graph_module.AgentRunError = AgentRunError
    graph_module.SessionBusyError = SessionBusyError
    graph_module.create_run = create_run
    graph_module.cancel_run = cancel_run
    graph_module.stream_agent_events = stream_agent_events
    monkeypatch.setitem(sys.modules, "app.agents.graph", graph_module)

    history_module = types.ModuleType("app.services.chat_history_service")

    async def save_chat_turn(turn: dict):
        saved_turns.append(turn)
        return turn

    history_module.save_chat_turn = save_chat_turn
    monkeypatch.setitem(sys.modules, "app.services.chat_history_service", history_module)

    memory_module = types.ModuleType("app.rag.chat_memory_store")

    async def save_chat_memory(turn: dict):
        saved_memories.append(turn)

    memory_module.save_chat_memory = save_chat_memory
    monkeypatch.setitem(sys.modules, "app.rag.chat_memory_store", memory_module)

    from app.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/chat/stream",
            json={
                "message": "我适合投什么岗位？",
                "session_id": "session-1",
                "user_profile": "统计学硕士，会 Python",
                "target_city": "北京",
            },
        )

    assert response.status_code == 200
    assert "event: status" in response.text
    assert "event: final" in response.text
    assert "建议优先投递数据分析师" in response.text

    assert len(saved_turns) == 1
    assert saved_turns[0]["status"] == "done"
    assert saved_turns[0]["meta"]["used_subagents"] == ["career-agent"]
    assert saved_turns[0]["meta"]["trace"]["trace_id"] == "run-1"
    assert saved_turns[0]["activity"]["trace"]["metrics"]["tool_call_count"] == 1
    assert saved_memories == saved_turns
