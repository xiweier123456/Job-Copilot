"""
app/services/resume_service.py
简历 - 岗位匹配分析服务。
流程：检索相关 JD → 拼接 prompt → LLM 分析 → 返回结构化结果
"""
import json
import re

from app.rag.embedder import embed_query
from app.rag.retriever import search_similar_jobs
from app.schemas.resume import ResumeMatchRequest, ResumeMatchResponse, SkillGap
from app.services.llm_client import chat_completion

SYSTEM_PROMPT = """\
你是一名专业的求职顾问，擅长分析简历与岗位要求的匹配度。
请根据用户提供的简历内容和参考岗位 JD，给出客观、具体、可操作的分析。
输出严格按照 JSON 格式，不要输出任何其他内容。
"""

MATCH_TEMPLATE = """\
## 简历内容
{resume}

## 参考岗位 JD（共 {jd_count} 条）
{jds}

## 分析要求
请对比简历与上述岗位要求，输出以下 JSON 结构（不含注释）：
{{
  "match_score": <0-100 的整数，表示整体匹配度>,
  "summary": "<2-3句话的整体评价>",
  "skill_gap": {{
    "matched": ["<匹配的技能/经历1>", "..."],
    "missing": ["<欠缺的技能/经历1>", "..."],
    "suggestions": ["<改进建议1>", "..."]
  }}
}}
"""


def _format_jds(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        salary = ""
        if h.get("min_salary") or h.get("max_salary"):
            salary = f"薪资：{int(h['min_salary'])}-{int(h['max_salary'])} 元/月，"
        parts.append(
            f"[JD {i}] {h['job_title']} | {h['company']} | {h['city']} | "
            f"{salary}学历：{h['education']}，经验：{h['experience']}\n"
            f"{h['text'][:600]}"
        )
    return "\n\n---\n\n".join(parts)


def _parse_llm_json(text: str) -> dict:
    """从 LLM 回复中提取 JSON，容错处理"""
    # 去掉 markdown code block
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # fallback：提取第一个 { ... } 块
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {}


async def analyze_resume_match(request: ResumeMatchRequest) -> ResumeMatchResponse:
    query_vector = embed_query(request.job_query)
    hits = search_similar_jobs(
        query_vector=query_vector,
        top_k=request.top_k,
        city=request.city,
        chunk_type="description",
    )

    if not hits:
        return ResumeMatchResponse(
            match_score=0,
            summary="未检索到相关岗位，请调整查询关键词。",
            skill_gap=SkillGap(),
            reference_jobs=[],
        )

    jd_text = _format_jds(hits)
    user_prompt = MATCH_TEMPLATE.format(
        resume=request.resume_text[:2000],
        jd_count=len(hits),
        jds=jd_text,
    )

    reply = await chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    parsed = _parse_llm_json(reply)

    skill_gap_raw = parsed.get("skill_gap", {})
    skill_gap = SkillGap(
        matched=skill_gap_raw.get("matched", []),
        missing=skill_gap_raw.get("missing", []),
        suggestions=skill_gap_raw.get("suggestions", []),
    )

    reference_jobs = [
        {
            "job_title": h["job_title"],
            "company":   h["company"],
            "city":      h["city"],
            "education": h["education"],
            "experience":h["experience"],
            "min_salary":h["min_salary"],
            "max_salary":h["max_salary"],
            "score":     h["score"],
        }
        for h in hits
    ]

    return ResumeMatchResponse(
        match_score=float(parsed.get("match_score", 0)),
        summary=parsed.get("summary", reply[:200]),
        skill_gap=skill_gap,
        reference_jobs=reference_jobs,
    )
