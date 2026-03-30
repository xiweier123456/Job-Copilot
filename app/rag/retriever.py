"""
app/rag/retriever.py
Milvus 向量检索封装，支持 scalar filter。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pymilvus import connections, Collection

from app.config import settings


@lru_cache(maxsize=1)
def _get_collection() -> Collection:
    connections.connect(host=settings.milvus_host, port=settings.milvus_port)
    col = Collection(settings.milvus_collection)
    col.load()
    return col


OUTPUT_FIELDS = [
    "chunk_id", "job_id", "chunk_type", "text",
    "company", "industry", "job_title", "city",
    "min_salary", "max_salary", "education", "experience",
    "publish_date", "year",
]


def search_similar_jobs(
    query_vector: list[float],
    top_k: int = 5,
    city: Optional[str] = None,
    industry: Optional[str] = None,
    education: Optional[str] = None,
    chunk_type: Optional[str] = None,   # "summary" | "description" | None(全部)
) -> list[dict]:
    """
    向量相似度检索，支持城市/行业/学历过滤。
    返回 list[dict]，每个 dict 包含 chunk 字段 + score。
    """
    col = _get_collection()

    # 构造 scalar filter 表达式
    filters = []
    if city:
        filters.append(f'city == "{city}"')
    if industry:
        filters.append(f'industry == "{industry}"')
    if education:
        filters.append(f'education == "{education}"')
    if chunk_type:
        filters.append(f'chunk_type == "{chunk_type}"')

    expr = " && ".join(filters) if filters else ""

    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 64},   # HNSW 检索参数
    }

    results = col.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        expr=expr or None,
        output_fields=OUTPUT_FIELDS,
    )

    hits = []
    for hit in results[0]:
        entity = hit.entity
        hits.append({
            "chunk_id":    entity.get("chunk_id", ""),
            "job_id":      entity.get("job_id", ""),
            "chunk_type":  entity.get("chunk_type", ""),
            "text":        entity.get("text", ""),
            "company":     entity.get("company", ""),
            "industry":    entity.get("industry", ""),
            "job_title":   entity.get("job_title", ""),
            "city":        entity.get("city", ""),
            "min_salary":  entity.get("min_salary", 0.0),
            "max_salary":  entity.get("max_salary", 0.0),
            "education":   entity.get("education", ""),
            "experience":  entity.get("experience", ""),
            "publish_date":entity.get("publish_date", ""),
            "year":        entity.get("year", 0),
            "score":       round(float(hit.score), 4),
        })
    return hits
