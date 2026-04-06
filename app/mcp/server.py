"""
app/mcp/server.py

求职 Copilot 的 FastMCP 服务入口。
这里从统一工具注册表读取 MCP 工具，确保 MCP 暴露层与 agent 工具层保持一致。
"""
import os
import sys

from fastmcp import FastMCP

from app.config import settings
from app.mcp.tool_registry import get_mcp_tools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

for mcp_tool in get_mcp_tools():
    mcp.tool()(mcp_tool)

if __name__ == "__main__":
    mcp.run(
        transport=MCP_TRANSPORT,
        host=MCP_HOST,
        port=MCP_PORT,
        path=MCP_PATH,
    )
