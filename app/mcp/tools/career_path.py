"""
app/mcp/tools/career_path.py
MCP tool：职业路径建议
"""
from app.rag.embedder import embed_query
from app.rag.retriever import search_similar_jobs
from app.services.llm_client import chat_completion


async def suggest_career_path(
    background: str,
    skills: str,
    target_city: str = "",
) -> dict:
    """
    根据学术背景和技能栈推荐适合的岗位方向和求职路径。

    Args:
        background: 学术背景，如"计算机硕士，研究方向自然语言处理"
        skills: 技能描述，如"Python, PyTorch, Transformer, SQL, Git"
        target_city: 目标城市，如"北京"（可选）

    Returns:
        推荐岗位方向、适合岗位列表、求职建议
    """
    # 用 background + skills 检索相关岗位
    query = f"{background} {skills}"
    query_vector = embed_query(query)
    hits = search_similar_jobs(
        query_vector=query_vector,
        top_k=5,
        city=target_city or None,
        chunk_type="description",
    )

    context = "\n\n".join([
        f"岗位：{h['job_title']} | 公司：{h['company']} | {h['city']}\n{h['text'][:400]}"
        for h in hits[:4]
    ])

    prompt = f"""\
学生背景：
{background}

掌握技能：
{skills}

目标城市：{target_city or '不限'}

参考岗位（从招聘数据库中检索）：
{context}

请根据以上信息给出：
1. 最适合该学生的 3 个岗位方向（每个方向给出岗位名称 + 推荐理由）
2. 每个方向的求职准备建议（需要补充什么技能/经历）
3. 整体求职路径建议
"""
    reply = await chat_completion(
        messages=[
            {"role": "system", "content": "你是一名专业的职业规划顾问，擅长帮助研究生制定求职路径。"},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.5,
        max_tokens=1200,
    )

    return {
        "tool": "suggest_career_path",
        "summary": "已根据用户背景和相似岗位生成职业路径建议。",
        "data": {
            "advice": reply,
            "reference_jobs": [
                {
                    "job_title": h["job_title"],
                    "company":   h["company"],
                    "city":      h["city"],
                    "education": h["education"],
                }
                for h in hits
            ],
        },
    }
