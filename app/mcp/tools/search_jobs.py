"""
app/mcp/tools/search_jobs.py

岗位语义检索的 MCP 工具封装。
这个模块保持 MCP 层足够薄，只负责参数接收和结果整理，真正的检索、
查询改写与重排逻辑下沉到 service 层。
"""
from typing import Optional

from app.services.job_service import search_jobs_service_with_meta


async def search_jobs(
    query: str,
    city: Optional[str] = None,
    industry: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """根据岗位、技能、城市或行业查询，返回结构化岗位证据。

    参数：
        query: 自然语言检索词，例如岗位名、技能组合或求职方向。
        city: 可选，按城市过滤岗位。
        industry: 可选，按行业过滤岗位。
        top_k: 最多返回多少条代表性岗位。

    返回：
        统一格式的结果字典，包含：
        - tool：稳定的工具名
        - summary：给调用方看的简短检索说明
        - data：代表性岗位列表
        - meta：底层检索服务返回的元信息
    """
    result = await search_jobs_service_with_meta(
        query=query,
        city=city,
        industry=industry,
        top_k=top_k,
    )
    items = [
        {
            "job_title": h.job_title,
            "company": h.company,
            "industry": h.industry,
            "city": h.city,
            "salary": f"{int(h.min_salary)}-{int(h.max_salary)} 元/月" if h.max_salary else "面议",
            "education": h.education,
            "experience": h.experience,
            "description": h.text[:400],
            "score": h.rerank_score if h.rerank_score is not None else h.score,
        }
        for h in result.items
    ]
    summary = f"共检索到 {len(items)} 个相关岗位结果。"
    if result.meta.rewritten:
        summary += f" 系统已自动进行了 {result.meta.attempt_count} 轮检索优化。"
    return {
        "tool": "search_jobs",
        "summary": summary,
        "data": items,
        "meta": result.meta.model_dump(),
    }
