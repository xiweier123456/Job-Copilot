from fastapi import APIRouter, Query
from typing import Optional

from app.schemas.job import JobSearchRequest, JobSearchResponse

router = APIRouter()


@router.get("/search", response_model=JobSearchResponse)
async def search_jobs(
    query: str = Query(..., description="查询文本"),
    city: Optional[str] = Query(None, description="城市"),
    industry: Optional[str] = Query(None, description="行业"),
    education: Optional[str] = Query(None, description="学历要求"),
    top_k: int = Query(5, ge=1, le=20, description="返回数量"),
):
    """
    语义检索相关岗位。
    - **query**: 关键词或描述，如 "数据分析师 SQL Python"
    - **city**: 城市过滤，如 "北京"
    - **industry**: 行业过滤，如 "银行"
    - **top_k**: 返回条数
    """
    from app.services.job_service import search_jobs_service
    results = await search_jobs_service(
        query=query, city=city, industry=industry,
        education=education, top_k=top_k
    )
    return JobSearchResponse(total=len(results), results=results)
