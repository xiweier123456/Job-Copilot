"""
app/agents/tools.py
供 deepagents 使用的工具定义。

设计原则：
- 工具描述必须具体，因为主 agent / subagent 会基于描述判断是否调用
- 工具实现尽量复用现有 MCP tools，而不是重复写一套业务逻辑
- 工具输出优先返回结构化证据，供 subagent 基于证据自行完成最终回答
"""
from langchain_core.tools import tool

from app.mcp.tools.search_jobs import search_jobs
from app.mcp.tools.tavily import (
    batch_tavily_search,
    tavily_extract,
    tavily_research,
    tavily_search,
)


@tool
async def search_jobs_tool(query: str, city: str = "", industry: str = "", top_k: int = 5) -> dict:
    """Fetch representative job evidence from the Milvus-backed job database.

    Use this tool when the user wants:
    - concrete job openings for a target role
    - jobs in a specific city or industry
    - examples of relevant postings
    - representative evidence about common requirements from real jobs

    Inputs:
    - query: role / skill / direction query text
    - city: optional city filter
    - industry: optional industry filter
    - top_k: number of representative postings to return

    Returns structured job evidence. Treat the output as source material, not as the final user-facing answer.
    """
    results = await search_jobs(
        query=query,
        city=city or None,
        industry=industry or None,
        top_k=top_k,
    )
    return results


@tool
async def tavily_search_tool(
    query: str,
    include_domains: str = "",
    exclude_domains: str = "",
    max_results: int = 5,
    topic: str = "general",
    time_range: str = "",
) -> dict:
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

    Returns structured search evidence from MCP-facing Tavily wrappers.
    """
    return await tavily_search(
        query=query,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        max_results=max_results,
        topic=topic,
        time_range=time_range,
    )


@tool
async def tavily_research_tool(
    query: str,
    include_domains: str = "",
    exclude_domains: str = "",
    max_results: int = 5,
) -> dict:
    """Run deeper Tavily research for multi-source synthesis without relying on CLI/bash.

    Use this tool when the user wants:
    - deeper research, comparisons, or trend summaries
    - synthesis across multiple sources rather than a quick lookup
    - market analysis or broad interview-prep takeaways from several sources

    Returns structured research evidence from MCP-facing Tavily wrappers.
    """
    return await tavily_research(
        query=query,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        max_results=max_results,
    )


@tool
async def tavily_extract_tool(urls: str) -> dict:
    """Extract clean page content from one or more URLs via Tavily.

    Use this tool when the user already has concrete URLs and you need page text for analysis.
    Input should be one or more URLs separated by commas or newlines.
    Returns structured extracted content from MCP-facing Tavily wrappers.
    """
    return await tavily_extract(urls=urls)


@tool
async def batch_tavily_search_tool(
    queries: str,
    max_results_per_query: int = 3,
    time_range: str = "",
) -> dict:
    """Run multiple Tavily searches concurrently to save time.

    Use this tool when you need to search multiple keywords at once.
    This is MUCH faster than calling tavily_search_tool multiple times in sequence.

    Inputs:
    - queries: multiple search queries separated by '|||'. Example: "query1 ||| query2 ||| query3"
    - max_results_per_query: number of results per query (default 3, keep small to reduce context)
    - time_range: optional time range filter (e.g., 'year', 'month')

    Returns structured batched search evidence from MCP-facing Tavily wrappers.
    """
    return await batch_tavily_search(
        queries=queries,
        max_results_per_query=max_results_per_query,
        time_range=time_range,
    )


ALL_TOOLS = [
    search_jobs_tool,
    tavily_search_tool,
    tavily_research_tool,
    tavily_extract_tool,
    batch_tavily_search_tool,
]
