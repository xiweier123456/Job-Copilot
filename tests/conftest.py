from __future__ import annotations

import os

import pytest

from app.schemas.job import JobChunk, RetrievalMeta, RetrievalResult


os.environ["DEBUG"] = "false"


def make_job_hit(**overrides) -> dict:
    """Build a complete Milvus-like hit dict for service-layer tests."""
    data = {
        "chunk_id": "chunk-1",
        "job_id": "job-1",
        "chunk_type": "description",
        "text": "负责数据分析、SQL 建模、业务指标体系建设。",
        "company": "示例科技",
        "industry": "互联网",
        "job_title": "数据分析师",
        "city": "北京",
        "min_salary": 12000.0,
        "max_salary": 22000.0,
        "education": "本科",
        "experience": "应届",
        "publish_date": "2026-03-01",
        "year": 2026,
        "score": 0.82,
        "rerank_score": None,
        "final_rank": None,
    }
    data.update(overrides)
    return data


@pytest.fixture
def sample_job_hit() -> dict:
    return make_job_hit()


@pytest.fixture
def sample_job_chunk(sample_job_hit: dict) -> JobChunk:
    return JobChunk(**sample_job_hit)


@pytest.fixture
def sample_retrieval_result(sample_job_chunk: JobChunk) -> RetrievalResult:
    return RetrievalResult(
        items=[sample_job_chunk],
        meta=RetrievalMeta(
            attempt_count=1,
            rewritten=False,
            final_query="数据分析师",
        ),
    )
