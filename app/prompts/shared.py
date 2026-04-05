"""Shared prompt fragments reused across agents and services."""

from __future__ import annotations

TAVILY_SKILL_PATHS = {
    "tavily-search": "skills/tavily/tavily-search/SKILL.md",
    "tavily-research": "skills/tavily/tavily-research/SKILL.md",
}

TAVILY_INSTRUCTIONS = (
    "\n<tavily_usage>\n"
    "## 联网搜索工具使用规范\n"
    "- 优先使用 batch_tavily_search_tool（多个关键词用 '|||' 分隔一次提交），效率最高\n"
    "- 单个搜索用 tavily_search_tool；深度多源归纳用 tavily_research_tool\n"
    "- 用户给出具体 URL 时，用 tavily_extract_tool 提取页面内容\n"
    f"- 如需读取 skill 文件，优先直接读取已知路径：\n"
    f"  - tavily-search → {TAVILY_SKILL_PATHS['tavily-search']}\n"
    f"  - tavily-research → {TAVILY_SKILL_PATHS['tavily-research']}\n"
    "  只有已知路径不可用时，才搜索 Tavily skill 目录\n"
    "- 如果 skill 中提到的搜索方式不可用，改用上述 Tavily Python 工具，不要建议用户手动执行命令\n"
    "- 如果工具返回 TAVILY_*_UNAVAILABLE，必须明确告知用户联网搜索不可用，不能假装搜索成功\n"
    "- 除非用户明确说不要上网搜索，否则必须先联网搜索真实信息再回答\n"
    "</tavily_usage>\n"
)
