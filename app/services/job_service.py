"""
app/services/job_service.py
岗位检索业务逻辑：query → embedding → Milvus 检索 → 结果封装
"""
from typing import Optional

from app.rag.embedder import embed_query
from app.rag.retriever import search_similar_jobs
from app.schemas.job import JobChunk


async def search_jobs_service(
    query: str,
    city: Optional[str] = None,
    industry: Optional[str] = None,
    education: Optional[str] = None,
    top_k: int = 5,
) -> list[JobChunk]:
    """
    用语义检索找相关岗位，优先返回 description chunk（包含完整 JD）。
    """
    query_vector = embed_query(query)

    hits = search_similar_jobs(
        query_vector=query_vector,
        top_k=top_k,
        city=city,
        industry=industry,
        education=education,
        chunk_type="description",   # 优先检索完整 JD
    )

    # 如果 description chunk 不够，补充 summary chunk
    if len(hits) < top_k:
        summary_hits = search_similar_jobs(
            query_vector=query_vector,
            top_k=top_k - len(hits),
            city=city,
            industry=industry,
            education=education,
            chunk_type="summary",
        )
        # 去重（同一 job_id 不重复出现）
        existing_job_ids = {h["job_id"] for h in hits}
        for h in summary_hits:
            if h["job_id"] not in existing_job_ids:
                hits.append(h)
                existing_job_ids.add(h["job_id"])

    return [JobChunk(**h) for h in hits]
