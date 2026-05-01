from __future__ import annotations

import pytest

from tests.conftest import make_job_hit


@pytest.mark.asyncio
async def test_retrieve_job_chunks_combines_summary_hits_and_dedupes(monkeypatch, sample_job_hit):
    """Retrieval should merge description/summary chunks and keep one result per job."""
    from app.services import retrieval_service

    monkeypatch.setattr(retrieval_service.settings, "enable_query_understanding", False)
    monkeypatch.setattr(retrieval_service.settings, "enable_retrieval_judge", False)
    monkeypatch.setattr(retrieval_service.settings, "enable_rerank", False)
    monkeypatch.setattr(retrieval_service, "embed_query", lambda query: [0.1] * 512)

    calls: list[dict] = []

    def fake_search_similar_jobs(**kwargs):
        calls.append(kwargs)
        if kwargs["chunk_type"] == "description":
            return [dict(sample_job_hit)]

        duplicate = make_job_hit(
            chunk_id="chunk-duplicate",
            chunk_type="summary",
            text="同一个 job_id 的 summary chunk，应被去重。",
        )
        another = make_job_hit(
            chunk_id="chunk-2",
            job_id="job-2",
            chunk_type="summary",
            job_title="商业分析师",
            company="样例咨询",
            score=0.77,
        )
        return [duplicate, another]

    def fake_rerank_hits(query, hits, top_k=None):
        ranked = []
        for index, hit in enumerate(hits[:top_k], start=1):
            item = dict(hit)
            item["rerank_score"] = round(0.9 - index * 0.01, 4)
            item["final_rank"] = index
            ranked.append(item)
        return ranked

    monkeypatch.setattr(retrieval_service, "search_similar_jobs", fake_search_similar_jobs)
    monkeypatch.setattr(retrieval_service, "rerank_hits", fake_rerank_hits)

    result = await retrieval_service.retrieve_job_chunks_with_meta(
        "数据分析 SQL",
        top_k=2,
        city="北京",
        industry="互联网",
    )

    assert [item.job_id for item in result.items] == ["job-1", "job-2"]
    assert result.items[0].final_rank == 1
    assert result.meta.attempt_count == 1
    assert result.meta.rewritten is False
    assert result.meta.final_query == "数据分析 SQL"

    assert [call["chunk_type"] for call in calls] == ["description", "summary"]
    assert all(call["city"] == "北京" for call in calls)
    assert all(call["industry"] == "互联网" for call in calls)
