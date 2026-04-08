"""
外部 MCP 服务统一注册与导入。

职责：
- 读取配置中的外部 MCP 服务定义
- 标准化为 FastMCP 的 MCPConfig 结构
- 建立 FastMCP Client 以发现远端工具
- 按 include / exclude / priority / prefix 生成可并入 registry 的规格
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastmcp import Client
from fastmcp.mcp_config import MCPConfig, RemoteMCPServer, StdioMCPServer

from app.config import ExternalMCPServiceConfig, settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExternalToolInfo:
    """描述一个外部 MCP 服务暴露出的单个工具。"""

    service_name: str
    source_name: str
    exposed_name: str
    description: str
    input_schema: dict[str, Any]
    priority: int
    requires_network: bool


@dataclass(frozen=True)
class ExternalServiceSnapshot:
    """描述一个外部 MCP 服务及其已发现的工具列表。"""

    service_name: str
    config: ExternalMCPServiceConfig
    tools: tuple[ExternalToolInfo, ...]


def _to_mcp_server_config(service: ExternalMCPServiceConfig) -> RemoteMCPServer | StdioMCPServer:
    """把项目内配置转换成 FastMCP 的标准服务配置。"""
    timeout = service.timeout or settings.mcp_client_timeout
    if service.transport == "stdio":
        return StdioMCPServer(
            command=service.command or "",
            args=service.args,
            env=service.env,
            cwd=service.cwd,
            timeout=timeout,
            keep_alive=service.keep_alive,
            description=service.description,
        )

    return RemoteMCPServer(
        url=service.url or "",
        transport=service.transport,
        headers=service.headers,
        timeout=timeout,
        description=service.description,
    )


def _build_mcp_config(service: ExternalMCPServiceConfig) -> MCPConfig:
    """为单个外部服务生成 FastMCP 可直接消费的 MCPConfig。"""
    config = MCPConfig()
    config.add_server(service.name, _to_mcp_server_config(service))
    return config


def _build_tool_name(service: ExternalMCPServiceConfig, source_name: str) -> str:
    """根据配置决定最终暴露给系统的工具名。"""
    if service.prefix:
        return f"{service.prefix}{source_name}"
    if settings.mcp_tool_name_prefix:
        return f"{service.name}_{source_name}"
    return source_name


def _is_tool_included(service: ExternalMCPServiceConfig, tool_name: str) -> bool:
    """按 include / exclude 规则判断是否纳入当前服务的工具。"""
    if service.include_tools and tool_name not in service.include_tools:
        return False
    if tool_name in service.exclude_tools:
        return False
    return True


async def discover_external_service_tools(
    service: ExternalMCPServiceConfig,
) -> ExternalServiceSnapshot:
    """连接单个外部 MCP 服务并发现其工具清单。"""
    config = _build_mcp_config(service)
    timeout = service.timeout or settings.mcp_client_timeout
    client = Client(config, name=f"external-mcp:{service.name}", timeout=timeout)

    async with client:
        remote_tools = await client.list_tools()

    tools: list[ExternalToolInfo] = []
    for tool in remote_tools:
        tool_name = getattr(tool, "name", "")
        if not tool_name or not _is_tool_included(service, tool_name):
            continue

        description = getattr(tool, "description", "") or ""
        input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
        exposed_name = _build_tool_name(service, tool_name)
        tools.append(
            ExternalToolInfo(
                service_name=service.name,
                source_name=tool_name,
                exposed_name=exposed_name,
                description=description,
                input_schema=input_schema,
                priority=service.priority,
                requires_network=service.transport != "stdio",
            )
        )

    return ExternalServiceSnapshot(
        service_name=service.name,
        config=service,
        tools=tuple(tools),
    )


async def discover_external_services() -> list[ExternalServiceSnapshot]:
    """发现当前启用的全部外部 MCP 服务。"""
    snapshots: list[ExternalServiceSnapshot] = []
    for service in settings.external_mcp_services:
        if not service.enabled:
            continue
        try:
            snapshots.append(await discover_external_service_tools(service))
        except Exception as exc:
            logger.warning("Failed to discover external MCP service %s: %s", service.name, exc)
    return snapshots


async def call_external_tool(
    service: ExternalMCPServiceConfig,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """调用指定外部 MCP 工具，并优先返回结构化结果。"""
    config = _build_mcp_config(service)
    timeout = service.timeout or settings.mcp_client_timeout
    client = Client(config, name=f"external-mcp:{service.name}", timeout=timeout)

    async with client:
        result = await client.call_tool(tool_name, arguments or {})

    if result.data is not None:
        return result.data
    if result.structured_content is not None:
        return result.structured_content
    return result.content
