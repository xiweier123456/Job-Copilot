"""
app/services/interview_service.py
求职问答 / 面试准备服务。
流程：检索相关岗位知识 → RAG → LLM 回答
"""
from app.rag.embedder import embed_query
from app.rag.retriever import search_similar_jobs
from app.services.llm_client import chat_completion

SYSTEM_PROMPT = """\
你是一名专业的求职顾问，帮助研究生和应届生解答求职相关问题。
你的回答应当具体、实用、有针对性，结合提供的岗位信息给出建议。
"""


def _format_context(hits: list[dict]) -> str:
    if not hits:
        return "（未找到相关岗位信息）"
    parts = []
    for h in hits[:3]:
        parts.append(
            f"岗位：{h['job_title']} | 公司：{h['company']} | 城市：{h['city']}\n"
            f"{h['text'][:500]}"
        )
    return "\n\n---\n\n".join(parts)


async def answer_career_question(message: str, session_id: str = "default") -> str:
    """
    基于 RAG 的求职问答。
    """
    # 检索相关岗位作为上下文
    query_vector = embed_query(message)
    hits = search_similar_jobs(query_vector=query_vector, top_k=3)
    context = _format_context(hits)

    user_prompt = f"""\
【参考岗位信息】
{context}

【用户问题】
{message}

请结合上述岗位信息回答用户问题。如果问题与岗位信息无直接关联，也可以根据通用求职知识回答。
"""

    reply = await chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=1000,
    )
    return reply
