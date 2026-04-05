"""
app/mcp/server.py
FastMCP server 入口。

运行方式（stdio 模式，供 MCP client 连接）：
  python -m app.mcp.server

或作为独立 HTTP server：
  python -m app.mcp.server --transport sse --port 8001
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from fastmcp import FastMCP
from app.mcp.tools.search_jobs import search_jobs
from app.mcp.tools.tavily import (
    batch_tavily_search,
    tavily_extract,
    tavily_research,
    tavily_search,
)
from app.config import settings
MCP_TRANSPORT = settings.mcp_transport
MCP_HOST = settings.mcp_host
MCP_PORT = settings.mcp_port
MCP_PATH = settings.mcp_path
mcp = FastMCP(
    name="job-copilot",
    instructions=(
        "求职助手 MCP Server，基于上市公司招聘数据（2024-2026）。\n"
        "当前提供岗位检索，以及 Tavily 联网搜索、深度研究、页面抽取能力。"
    ),
)

# 注册所有 tools
mcp.tool()(search_jobs)
mcp.tool()(tavily_search)
mcp.tool()(tavily_research)
mcp.tool()(tavily_extract)
mcp.tool()(batch_tavily_search)

if __name__ == "__main__":
    mcp.run(
        transport=MCP_TRANSPORT,
        host=MCP_HOST,
        port=MCP_PORT,
        path=MCP_PATH,
    )
