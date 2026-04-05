"""
app/services/interview_service.py
求职问答 / 面试准备服务。
流程：检索相关岗位知识 → RAG → Rerank 后 LLM 回答
"""
from app.prompts.service_prompts import build_interview_answer_messages
from app.services.llm_client import chat_completion
from app.services.retrieval_service import retrieve_job_chunks


async def answer_career_question(message: str, session_id: str = "default") -> str:
    """
    基于 RAG 的求职问答。
    """
    hits = [item.model_dump() for item in await retrieve_job_chunks(message, top_k=3)]
    reply = await chat_completion(
        messages=build_interview_answer_messages(message=message, hits=hits),
        temperature=0.5,
        max_tokens=1000,
    )
    return reply
