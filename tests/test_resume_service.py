from __future__ import annotations

import json

import pytest

from app.schemas.resume import ResumeMatchRequest


@pytest.mark.asyncio
async def test_analyze_resume_match_uses_retrieved_jobs_and_parses_llm_json(
    monkeypatch,
    sample_job_chunk,
):
    """Resume matching should combine retrieved JD evidence with the LLM JSON result."""
    from app.services import resume_service

    async def fake_retrieve_job_chunks(query, *, top_k, city=None):
        assert query == "数据分析师"
        assert top_k == 3
        assert city == "北京"
        return [sample_job_chunk]

    async def fake_chat_completion(messages, temperature, max_tokens):
        assert temperature == 0.2
        assert max_tokens == 1200
        joined = "\n".join(message["content"] for message in messages)
        assert "Python SQL 项目" in joined
        assert "数据分析师" in joined
        return json.dumps(
            {
                "match_score": 82,
                "summary": "匹配度较高，SQL 和项目表达仍可加强。",
                "skill_gap": {
                    "matched": ["Python", "SQL"],
                    "missing": ["BI 可视化"],
                    "suggestions": ["补充业务指标分析项目"],
                },
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(resume_service, "retrieve_job_chunks", fake_retrieve_job_chunks)
    monkeypatch.setattr(resume_service, "chat_completion", fake_chat_completion)

    response = await resume_service.analyze_resume_match(
        ResumeMatchRequest(
            resume_text="Python SQL 项目，做过用户留存分析。",
            job_query="数据分析师",
            city="北京",
            top_k=3,
        )
    )

    assert response.match_score == 82
    assert response.skill_gap.matched == ["Python", "SQL"]
    assert response.skill_gap.missing == ["BI 可视化"]
    assert response.reference_jobs[0]["job_title"] == sample_job_chunk.job_title


@pytest.mark.asyncio
async def test_analyze_resume_match_returns_empty_result_when_no_jobs(monkeypatch):
    """An empty retrieval result should not call the LLM or pretend to have evidence."""
    from app.services import resume_service

    async def fake_retrieve_job_chunks(*args, **kwargs):
        return []

    async def fail_if_called(*args, **kwargs):  # pragma: no cover - should never run
        raise AssertionError("LLM should not be called when there are no reference jobs")

    monkeypatch.setattr(resume_service, "retrieve_job_chunks", fake_retrieve_job_chunks)
    monkeypatch.setattr(resume_service, "chat_completion", fail_if_called)

    response = await resume_service.analyze_resume_match(
        ResumeMatchRequest(
            resume_text="简历文本",
            job_query="不存在的岗位",
            top_k=3,
        )
    )

    assert response.match_score == 0
    assert response.reference_jobs == []
    assert "未检索到相关岗位" in response.summary
