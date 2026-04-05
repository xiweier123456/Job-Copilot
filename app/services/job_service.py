"""
app/services/job_service.py
岗位检索业务逻辑：query → embedding → Milvus 检索 → rerank → 结果封装
"""
from typing import Optional

from app.schemas.job import JobChunk, RetrievalResult
from app.services.retrieval_service import retrieve_job_chunks, retrieve_job_chunks_with_meta


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
    return await retrieve_job_chunks(
        query,
        top_k=top_k,
        city=city,
        industry=industry,
        education=education,
    )


async def search_jobs_service_with_meta(
    query: str,
    city: Optional[str] = None,
    industry: Optional[str] = None,
    education: Optional[str] = None,
    top_k: int = 5,
) -> RetrievalResult:
    return await retrieve_job_chunks_with_meta(
        query,
        top_k=top_k,
        city=city,
        industry=industry,
        education=education,
    )
