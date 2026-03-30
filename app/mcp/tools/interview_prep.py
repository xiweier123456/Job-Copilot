"""
app/mcp/tools/interview_prep.py
MCP tool：面试准备建议生成
"""
from app.rag.embedder import embed_query
from app.rag.retriever import search_similar_jobs
from app.services.llm_client import chat_completion


async def generate_interview_prep(
    job_title: str,
    background: str = "",
) -> dict:
    """
    根据目标岗位生成面试常见问题和准备建议。

    Args:
        job_title: 目标岗位名称，如"数据分析师"、"算法工程师"
        background: 应聘者背景（可选），如"计算机硕士，NLP 方向"

    Returns:
        常见面试问题列表、技术准备建议、行为面试建议
    """
    # 检索相关 JD，提取岗位要求作为上下文
    query_vector = embed_query(job_title)
    hits = search_similar_jobs(
        query_vector=query_vector,
        top_k=3,
        chunk_type="description",
    )

    jd_context = "\n\n".join([
        f"[JD {i+1}] {h['job_title']} @ {h['company']}（{h['city']}）\n{h['text'][:500]}"
        for i, h in enumerate(hits)
    ])

    prompt = f"""\
目标岗位：{job_title}
应聘者背景：{background or '研究生/应届生'}

参考 JD（来自真实招聘数据）：
{jd_context}

请生成以下内容：
1. 该岗位 5-8 个常见面试问题（包含技术题和行为题）
2. 技术准备建议（需要重点准备的知识点/工具）
3. 行为面试准备建议（如何展示项目经历、团队协作等）
4. 简历投递前的注意事项

请用条目清单格式输出，便于阅读。
"""
    reply = await chat_completion(
        messages=[
            {"role": "system", "content": "你是一名有 5 年经验的技术面试官兼求职顾问，熟悉各类企业的招聘流程。"},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.4,
        max_tokens=1500,
    )

    return {
        "tool": "generate_interview_prep",
        "summary": f"已为目标岗位 {job_title} 生成面试准备建议。",
        "data": {
            "job_title": job_title,
            "interview_prep": reply,
            "reference_jobs": [
                {"job_title": h["job_title"], "company": h["company"], "city": h["city"]}
                for h in hits
            ],
        },
    }
