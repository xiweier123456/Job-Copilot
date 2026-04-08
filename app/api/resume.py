from fastapi import APIRouter, File, Form, UploadFile
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


@router.post("/match/upload", response_model=ResumeMatchResponse)
async def match_resume_upload(
    resume_file: UploadFile = File(..., description="上传的 PDF 简历"),
    job_query: str = Form(..., description="目标岗位名称或描述"),
    city: str = Form(default="", description="目标城市，可选"),
    top_k: int = Form(default=3, description="参考岗位数量"),
):
    """上传 PDF 简历并复用现有简历匹配流程。"""
    from app.services.document_service import extract_text_from_pdf_upload
    from app.services.resume_service import analyze_resume_match

    resume_text = await extract_text_from_pdf_upload(resume_file)
    request = ResumeMatchRequest(
        resume_text=resume_text,
        job_query=job_query,
        city=city or None,
        top_k=top_k,
    )
    return await analyze_resume_match(request)
