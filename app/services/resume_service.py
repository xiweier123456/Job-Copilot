"""
app/services/resume_service.py
简历 - 岗位匹配分析服务。
流程：检索相关 JD → LLM 分析 → 返回结构化结果
"""

from app.prompts.service_prompts import build_resume_match_messages
from app.schemas.resume import ResumeMatchRequest, ResumeMatchResponse, SkillGap
from app.services.llm_client import chat_completion, parse_json_response
from app.services.retrieval_service import retrieve_job_chunks


def _build_reference_jobs(hits: list[dict]) -> list[dict]:
    return [
        {
            "job_title": h["job_title"],
            "company": h["company"],
            "city": h["city"],
            "education": h["education"],
            "experience": h["experience"],
            "min_salary": h["min_salary"],
            "max_salary": h["max_salary"],
            "score": h["score"],
        }
        for h in hits
    ]


async def analyze_resume_match(request: ResumeMatchRequest) -> ResumeMatchResponse:
    hits = [
        item.model_dump()
        for item in await retrieve_job_chunks(
            request.job_query,
            top_k=request.top_k,
            city=request.city,
        )
    ]

    if not hits:
        return ResumeMatchResponse(
            match_score=0,
            summary="未检索到相关岗位，请调整查询关键词。",
            skill_gap=SkillGap(),
            reference_jobs=[],
        )

    reply = await chat_completion(
        messages=build_resume_match_messages(
            resume=request.resume_text[:2000],
            hits=hits,
        ),
        temperature=0.2,
        max_tokens=1200,
    )

    parsed = parse_json_response(reply)

    skill_gap_raw = parsed.get("skill_gap", {})
    skill_gap = SkillGap(
        matched=skill_gap_raw.get("matched", []),
        missing=skill_gap_raw.get("missing", []),
        suggestions=skill_gap_raw.get("suggestions", []),
    )

    return ResumeMatchResponse(
        match_score=float(parsed.get("match_score", 0)),
        summary=parsed.get("summary", reply[:200]),
        skill_gap=skill_gap,
        reference_jobs=_build_reference_jobs(hits),
    )
