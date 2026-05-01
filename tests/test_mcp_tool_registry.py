from __future__ import annotations


def test_local_tool_registry_exposes_metadata_without_external_discovery(monkeypatch):
    """Local tools should expose stable metadata for API, SSE, and frontend display."""
    from app.mcp import tool_registry

    monkeypatch.setattr(tool_registry, "_get_cached_external_tool_specs", lambda force_refresh=False: ())

    specs = tool_registry.get_tool_specs()
    search_spec = tool_registry.get_tool_spec("search_jobs")
    serialized = tool_registry.serialize_tool_spec(search_spec, status="started")

    assert search_spec in specs
    assert serialized["name"] == "search_jobs_tool"
    assert serialized["category"] == "job_db"
    assert serialized["requires_network"] is False
    assert serialized["status"] == "started"

    interview_tool_keys = {spec.key for spec in tool_registry.get_subagent_tool_specs("interview-agent")}
    assert {"search_jobs", "tavily_search", "tavily_research", "tavily_extract"}.issubset(interview_tool_keys)


def test_external_mcp_tools_are_merged_into_subagent_tool_specs(monkeypatch):
    """External MCP tools should be available to subagents through the same registry API."""
    from app.mcp import tool_registry

    async def fake_tool(**kwargs):
        return kwargs

    external_spec = tool_registry.ToolSpec(
        key="external::remote-search::web_search",
        agent_name="remote_web_search",
        display_name="remote_web_search",
        description="Remote MCP web search tool",
        category="external_mcp",
        requires_network=True,
        latency="medium",
        evidence_type="external_mcp",
        enabled=True,
        raw_callable=fake_tool,
        agent_callable=fake_tool,
        source_type="external_mcp",
        source_name="remote-search",
        canonical_name="remote_web_search:10",
        raw_name="web_search",
        exposed_name="remote_web_search",
    )

    monkeypatch.setattr(
        tool_registry,
        "_get_cached_external_tool_specs",
        lambda force_refresh=False: (external_spec,),
    )

    specs = tool_registry.get_subagent_tool_specs("career-agent")

    assert external_spec in specs
    assert tool_registry.get_tool_spec_by_agent_name("remote_web_search") == external_spec
