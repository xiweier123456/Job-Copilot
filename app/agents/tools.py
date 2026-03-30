"""
app/agents/tools.py
供 deepagents 使用的工具定义。

设计原则：
- 工具描述必须具体，因为主 agent / subagent 会基于描述判断是否调用
- 工具实现尽量复用现有 MCP tools，而不是重复写一套业务逻辑
- 返回值优先保持可读、结构清晰，方便 agent 直接拼接最终回答
"""
from langchain_core.tools import tool

from app.mcp.tools.search_jobs import search_jobs
from app.mcp.tools.resume_match import analyze_resume_match
from app.mcp.tools.career_path import suggest_career_path
from app.mcp.tools.interview_prep import generate_interview_prep
from app.services.tavily_client import (
    TavilyError,
    tavily_extract,
    tavily_research,
    tavily_search,
)


@tool
async def search_jobs_tool(query: str, city: str = "", industry: str = "", top_k: int = 5) -> str:
    """Search real job postings from the Milvus-backed job database.

    Use this tool when the user wants:
    - concrete job openings for a target role
    - jobs in a specific city or industry
    - examples of relevant postings
    - a summary of common requirements based on real retrieved jobs

    Best for questions like:
    - "深圳有哪些数据分析岗？"
    - "算法工程师一般要求什么？"
    - "帮我找几个适合研究生的 AI 岗位"

    Inputs:
    - query: role / skill / direction query text
    - city: optional city filter
    - industry: optional industry filter
    - top_k: number of representative postings to return

    Returns a readable summary of retrieved jobs including title, company, city, salary, education, experience, and description snippet.
    """
    results = await search_jobs(
        query=query,
        city=city or None,
        industry=industry or None,
        top_k=top_k,
    )
    items = results["data"]
    if not items:
        return "未找到相关岗位。"

    lines = []
    for item in items:
        lines.append(
            f"【{item['job_title']}】{item['company']} | {item['city']} | {item['salary']} | 学历：{item['education']} | 经验：{item['experience']}\n"
            f"{item['description']}"
        )
    return "\n\n---\n\n".join(lines)


@tool
async def analyze_resume_tool(resume_text: str, job_query: str, city: str = "") -> str:
    """Evaluate how well a resume matches a target role using real retrieved job descriptions.

    Use this tool when the user:
    - pastes a resume and asks whether it matches a role
    - wants strengths / weaknesses / skill gaps
    - asks how to improve a resume for a target position

    Best for questions like:
    - "这是我的简历，适合数据分析师吗？"
    - "我投算法岗还缺什么？"
    - "帮我看看这份简历怎么改"

    Inputs:
    - resume_text: the user's resume text
    - job_query: target role or role direction
    - city: optional target city filter

    Returns a readable analysis including match score, summary, matched skills, missing skills, and actionable suggestions.
    """
    result = await analyze_resume_match(
        resume_text=resume_text,
        job_query=job_query,
        city=city or None,
    )
    data = result["data"]
    return (
        f"匹配度：{data['match_score']}\n"
        f"整体评价：{data['summary']}\n"
        f"匹配技能：{', '.join(data['matched_skills']) or '无'}\n"
        f"缺失技能：{', '.join(data['missing_skills']) or '无'}\n"
        f"建议：{'；'.join(data['suggestions']) or '无'}"
    )


@tool
async def suggest_career_path_tool(background: str, skills: str = "", target_city: str = "") -> str:
    """Recommend realistic job directions and preparation paths for a graduate student or new graduate.

    Use this tool when the user wants:
    - role-direction advice
    - help choosing between several job paths
    - short-term preparation planning based on background and skills

    Best for questions like:
    - "我适合投什么岗位？"
    - "我是计算机硕士，接下来该怎么准备求职？"
    - "NLP 方向研究生适合哪些岗位？"

    Inputs:
    - background: degree / major / research / project background
    - skills: current skill stack
    - target_city: optional preferred city

    Returns practical career-path advice grounded in similar retrieved jobs.
    """
    result = await suggest_career_path(
        background=background,
        skills=skills,
        target_city=target_city,
    )
    return result["data"]["advice"]


@tool
async def interview_prep_tool(job_title: str, background: str = "") -> str:
    """Generate interview preparation advice and likely interview questions for a target role.

    Use this tool when the user wants:
    - likely interview questions
    - preparation priorities for a role
    - project / technical / behavioral interview guidance
    - a pre-interview or pre-submission checklist

    Best for questions like:
    - "算法工程师面试怎么准备？"
    - "数据分析师常见面试题有哪些？"
    - "帮我准备这个岗位的面试"

    Inputs:
    - job_title: target role title
    - background: optional candidate background

    Returns interview questions, answer directions, preparation focus, and checklist items.
    """
    result = await generate_interview_prep(
        job_title=job_title,
        background=background,
    )
    return result["data"]["interview_prep"]


@tool
async def tavily_search_tool(
    query: str,
    include_domains: str = "",
    exclude_domains: str = "",
    max_results: int = 5,
    topic: str = "general",
    time_range: str = "",
) -> str:
    """Search current web sources via Tavily without using shell commands.

    Use this tool when the user wants:
    - recent or real-world information from the web
    - articles, posts, company pages, or market signals
    - online interview experiences from sites like 牛客网 or 知乎
    - grounded web results before giving a job-search / resume / interview answer

    Best for questions like:
    - "帮我搜一下 2026 算法工程师校招面经"
    - "找一些牛客和知乎上的真实经验"
    - "看看这个岗位最近的真实要求"

    Returns a readable summary of search results with titles, URLs, snippets, and any answer field Tavily provides.
    If Tavily is unavailable, returns an explicit failure message instead of pretending web search succeeded.
    """
    try:
        result = await tavily_search(
            query=query,
            max_results=max_results,
            topic=topic,
            time_range=time_range or None,
            include_domains=[d.strip() for d in include_domains.split(",") if d.strip()] or None,
            exclude_domains=[d.strip() for d in exclude_domains.split(",") if d.strip()] or None,
            include_answer=True,
        )
    except TavilyError as exc:
        return f"TAVILY_SEARCH_UNAVAILABLE: {exc}"

    lines = []
    answer = result.get("answer")
    if isinstance(answer, str) and answer.strip():
        lines.append(f"总结：{answer.strip()}")

    items = result.get("results") or []
    if not items:
        lines.append("未检索到相关网络结果。")
        return "\n\n".join(lines)

    lines.append(f"共检索到 {len(items)} 条网络结果：")
    for idx, item in enumerate(items[:max_results], start=1):
        title = item.get("title") or "无标题"
        url = item.get("url") or "无链接"
        content = (item.get("content") or "").strip()
        score = item.get("score")
        snippet = content[:240] + ("..." if len(content) > 240 else "")
        score_text = f" | 相关度：{score}" if score is not None else ""
        lines.append(f"{idx}. {title}{score_text}\n链接：{url}\n摘要：{snippet or '无摘要'}")

    return "\n\n".join(lines)


@tool
async def tavily_research_tool(
    query: str,
    include_domains: str = "",
    exclude_domains: str = "",
    max_results: int = 5,
) -> str:
    """Run deeper Tavily research for multi-source synthesis without relying on CLI/bash.

    Use this tool when the user wants:
    - deeper research, comparisons, or trend summaries
    - synthesis across multiple sources rather than a quick lookup
    - market analysis or broad interview-prep takeaways from several sources

    Returns a concise research summary and source list.
    If Tavily is unavailable, returns an explicit failure message instead of pretending research succeeded.
    """
    try:
        result = await tavily_research(
            query=query,
            max_results=max_results,
            include_domains=[d.strip() for d in include_domains.split(",") if d.strip()] or None,
            exclude_domains=[d.strip() for d in exclude_domains.split(",") if d.strip()] or None,
        )
    except TavilyError as exc:
        return f"TAVILY_RESEARCH_UNAVAILABLE: {exc}"

    lines = []
    for key in ("summary", "answer", "report"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(value.strip())
            break

    items = result.get("results") or result.get("sources") or []
    if items:
        lines.append("参考来源：")
        for idx, item in enumerate(items[:max_results], start=1):
            title = item.get("title") or "无标题"
            url = item.get("url") or "无链接"
            lines.append(f"{idx}. {title} | {url}")
    else:
        lines.append("未返回可用来源。")

    return "\n\n".join(lines)


@tool
async def tavily_extract_tool(urls: str) -> str:
    """Extract clean page content from one or more URLs via Tavily.

    Use this tool when the user already has concrete URLs and you need page text for analysis.
    Input should be one or more URLs separated by commas or newlines.
    Returns extracted markdown/text snippets. If Tavily is unavailable, returns an explicit failure message.
    """
    url_list = [item.strip() for item in urls.replace("\n", ",").split(",") if item.strip()]
    if not url_list:
        return "未提供可提取的 URL。"

    try:
        result = await tavily_extract(url_list)
    except TavilyError as exc:
        return f"TAVILY_EXTRACT_UNAVAILABLE: {exc}"

    items = result.get("results") or result.get("data") or []
    if not items:
        return "未提取到页面内容。"

    lines = []
    for idx, item in enumerate(items[:5], start=1):
        url = item.get("url") or url_list[min(idx - 1, len(url_list) - 1)]
        raw = (item.get("raw_content") or item.get("content") or "").strip()
        snippet = raw[:800] + ("..." if len(raw) > 800 else "")
        lines.append(f"{idx}. {url}\n内容：{snippet or '无内容'}")
    return "\n\n".join(lines)


@tool
async def batch_tavily_search_tool(
    queries: str,
    max_results_per_query: int = 3,
    time_range: str = "",
) -> str:
    """Run multiple Tavily searches concurrently to save time.

    Use this tool when you need to search multiple keywords at once.
    This is MUCH faster than calling tavily_search_tool multiple times in sequence.

    Inputs:
    - queries: multiple search queries separated by '|||'. Example: "query1 ||| query2 ||| query3"
    - max_results_per_query: number of results per query (default 3, keep small to reduce context)
    - time_range: optional time range filter (e.g., 'year', 'month')

    Returns combined results from all queries, each section labeled with its query.
    """
    import asyncio

    query_list = [q.strip() for q in queries.split("|||") if q.strip()]
    if not query_list:
        return "未提供搜索关键词。"

    async def _single_search(query: str) -> tuple[str, str]:
        try:
            result = await tavily_search(
                query=query,
                max_results=max_results_per_query,
                time_range=time_range or None,
                include_answer=True,
            )
        except TavilyError as exc:
            return query, f"搜索失败：{exc}"

        lines = []
        answer = result.get("answer")
        if isinstance(answer, str) and answer.strip():
            lines.append(f"摘要：{answer.strip()[:200]}")

        items = result.get("results") or []
        for idx, item in enumerate(items[:max_results_per_query], start=1):
            title = item.get("title") or "无标题"
            url = item.get("url") or ""
            content = (item.get("content") or "").strip()
            snippet = content[:150] + ("..." if len(content) > 150 else "")
            lines.append(f"{idx}. {title}\n   链接：{url}\n   摘要：{snippet or '无摘要'}")

        return query, "\n".join(lines) if lines else "未检索到结果。"

    results = await asyncio.gather(*[_single_search(q) for q in query_list])

    output_lines = []
    for query, result_text in results:
        output_lines.append(f"### 搜索：{query}\n{result_text}")

    return "\n\n---\n\n".join(output_lines)


ALL_TOOLS = [
    search_jobs_tool,
    analyze_resume_tool,
    suggest_career_path_tool,
    interview_prep_tool,
    tavily_search_tool,
    tavily_research_tool,
    tavily_extract_tool,
    batch_tavily_search_tool,
]
