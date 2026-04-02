"""
app/mcp/tools/search_jobs.py
MCP tool：语义检索相关岗位
"""
from typing import Optional
import asyncio

from app.rag.embedder import embed_query
from app.rag.retriever import search_similar_jobs


async def search_jobs(
    query: str,
    city: Optional[str] = None,
    industry: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    根据查询文本语义检索相关岗位。

    Args:
        query: 查询文本，如"数据分析师 SQL Python"或"硕士 AI 算法北京"
        city: 城市过滤，如"北京"、"上海"（可选）
        industry: 行业过滤，如"银行"、"科技"（可选）
        top_k: 返回结果数量，默认 5

    Returns:
        匹配岗位列表，每条包含岗位名、公司、城市、薪资、JD 摘要、相似度得分
    """
    query_vector = embed_query(query)# 将查询文本转换为向量
    hits = search_similar_jobs(
        query_vector=query_vector,
        top_k=top_k,
        city=city,
        industry=industry,
        chunk_type="description",
    )
    items = [
        {
            "job_title":   h["job_title"],
            "company":     h["company"],
            "industry":    h["industry"],
            "city":        h["city"],
            "salary":      f"{int(h['min_salary'])}-{int(h['max_salary'])} 元/月" if h["max_salary"] else "面议",
            "education":   h["education"],
            "experience":  h["experience"],
            "description": h["text"][:400],
            "score":       h["score"],
        }
        for h in hits
    ]
    return {
        "tool": "search_jobs",
        "summary": f"共检索到 {len(items)} 个相关岗位结果。",
        "data": items,
    }
