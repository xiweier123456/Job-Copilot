"""Prompt builders for direct service-level LLM calls."""

from __future__ import annotations

from app.prompts.helpers import join_sections


CONTEXT_COMPRESSION_SYSTEM_PROMPT = """
你是对话上下文压缩器。你的任务是把历史对话和长期记忆压缩成紧凑但可用的上下文，供求职 Agent 继续回答当前问题。

要求：
- 只保留会影响当前回答的事实、用户偏好、简历背景、目标城市、目标岗位、已给过的重要建议、工具证据和待办结论。
- 删除寒暄、重复表达、无关网页噪声、失败尝试和冗余格式。
- 不要改写当前用户问题；当前用户问题会由调用方原样拼接。
- 输出中文。
- 只返回 JSON，不要 markdown，不要解释。

JSON 格式：
{
  "compressed_context": "压缩后的上下文",
  "kept_facts": ["保留下来的关键事实"]
}
""".strip()


def build_context_compression_messages(
    *,
    current_message: str,
    supplemental_context: str,
    target_tokens: int,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CONTEXT_COMPRESSION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": join_sections(
                "请压缩下面的补充上下文。",
                f"目标 token 数约为：{target_tokens}",
                f"【当前用户问题，仅用于判断相关性，不要改写】\n{current_message}",
                f"【待压缩补充上下文】\n{supplemental_context}",
            ),
        },
    ]


RESUME_MATCH_SYSTEM_PROMPT = """\
你是一名专业的求职顾问，擅长分析简历与岗位要求的匹配度。
请根据用户提供的简历内容和参考岗位 JD，给出客观、具体、可操作的分析。
输出严格按照 JSON 格式，不要输出任何其他内容。
"""


def format_resume_jds(hits: list[dict], *, limit: int | None = None, text_limit: int = 600) -> str:
    parts = []
    selected_hits = hits if limit is None else hits[:limit]
    for i, h in enumerate(selected_hits, 1):
        salary = ""
        if h.get("min_salary") or h.get("max_salary"):
            salary = f"薪资：{int(h['min_salary'])}-{int(h['max_salary'])} 元/月，"
        parts.append(
            f"[JD {i}] {h['job_title']} | {h['company']} | {h['city']} | "
            f"{salary}学历：{h['education']}，经验：{h['experience']}\n"
            f"{h['text'][:text_limit]}"
        )
    return "\n\n---\n\n".join(parts)


def build_resume_match_user_prompt(resume: str, hits: list[dict]) -> str:
    return join_sections(
        "## 简历内容\n" + resume,
        f"## 参考岗位 JD（共 {len(hits)} 条）\n{format_resume_jds(hits)}",
        "## 分析要求\n"
        "请对比简历与上述岗位要求，输出以下 JSON 结构（不含注释）：\n"
        "{\n"
        '  "match_score": <0-100 的整数，表示整体匹配度>,\n'
        '  "summary": "<2-3句话的整体评价>",\n'
        '  "skill_gap": {\n'
        '    "matched": ["<匹配的技能/经历1>", "..."],\n'
        '    "missing": ["<欠缺的技能/经历1>", "..."],\n'
        '    "suggestions": ["<改进建议1>", "..."]\n'
        "  }\n"
        "}",
    )


def build_resume_match_messages(resume: str, hits: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RESUME_MATCH_SYSTEM_PROMPT},
        {"role": "user", "content": build_resume_match_user_prompt(resume=resume, hits=hits)},
    ]


INTERVIEW_SYSTEM_PROMPT = """\
你是一名专业的求职顾问，帮助研究生和应届生解答求职相关问题。
你的回答应当具体、实用、有针对性，结合提供的岗位信息给出建议。
"""


def format_interview_context(hits: list[dict]) -> str:
    if not hits:
        return "（未找到相关岗位信息）"
    parts = []
    for h in hits[:3]:
        parts.append(
            f"岗位：{h['job_title']} | 公司：{h['company']} | 城市：{h['city']}\n"
            f"{h['text'][:500]}"
        )
    return "\n\n---\n\n".join(parts)


def build_interview_answer_user_prompt(message: str, hits: list[dict]) -> str:
    return join_sections(
        f"【参考岗位信息】\n{format_interview_context(hits)}",
        f"【用户问题】\n{message}",
        "请结合上述岗位信息回答用户问题。如果问题与岗位信息无直接关联，也可以根据通用求职知识回答。",
    )


def build_interview_answer_messages(message: str, hits: list[dict]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": build_interview_answer_user_prompt(message=message, hits=hits)},
    ]
