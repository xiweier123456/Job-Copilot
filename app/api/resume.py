from fastapi import APIRouter
from app.schemas.resume import ResumeMatchRequest, ResumeMatchResponse

router = APIRouter()


@router.post("/match", response_model=ResumeMatchResponse)
async def match_resume(request: ResumeMatchRequest):
    """
    简历与目标岗位匹配分析。
    - 自动检索相关 JD
    - LLM 分析技能匹配度与缺口
    - 给出改进建议
    """
    from app.services.resume_service import analyze_resume_match
    return await analyze_resume_match(request)
