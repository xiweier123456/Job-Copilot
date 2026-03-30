"""
app/mcp/tools/resume_match.py
MCP tool：简历 - 岗位匹配分析
"""
from typing import Optional

from app.services.resume_service import analyze_resume_match as _analyze
from app.schemas.resume import ResumeMatchRequest


async def analyze_resume_match(
    resume_text: str,
    job_query: str,
    city: Optional[str] = None,
) -> dict:
    """
    分析简历与目标岗位的匹配度，给出技能缺口和改进建议。

    Args:
        resume_text: 简历文本内容（粘贴纯文本即可）
        job_query: 目标岗位名称或描述，如"数据分析师"或"Python 后端开发"
        city: 目标城市（可选）

    Returns:
        匹配度评分、整体评价、技能缺口分析、改进建议、参考岗位列表
    """
    req = ResumeMatchRequest(
        resume_text=resume_text,
        job_query=job_query,
        city=city,
        top_k=3,
    )
    result = await _analyze(req)
    return {
        "tool": "analyze_resume_match",
        "summary": f"已完成简历与目标岗位的匹配分析，匹配度为 {result.match_score}。",
        "data": {
            "match_score": result.match_score,
            "summary": result.summary,
            "matched_skills": result.skill_gap.matched,
            "missing_skills": result.skill_gap.missing,
            "suggestions": result.skill_gap.suggestions,
            "reference_jobs": result.reference_jobs,
        },
    }
