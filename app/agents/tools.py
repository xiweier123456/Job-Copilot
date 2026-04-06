"""
app/agents/tools.py

deep agent 工具 wrapper 的兼容导出层。
实际的逻辑工具目录维护在 app.mcp.tool_registry 中，这里只保留稳定导出，
避免外部 import 路径在重构后失效。
"""
from app.mcp.tool_registry import (
    batch_tavily_search_tool as batch_tavily_search_tool,
    get_all_agent_tools,
    search_jobs_tool as search_jobs_tool,
    tavily_extract_tool as tavily_extract_tool,
    tavily_research_tool as tavily_research_tool,
    tavily_search_tool as tavily_search_tool,
)


ALL_TOOLS = get_all_agent_tools()
