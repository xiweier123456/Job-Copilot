from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    agent_model_provider: str | None = Field(default=None, description="Agent model provider, such as minimax or deepseek")
    message: str = Field(..., description="User message")
    session_id: str = Field(default="default", description="Conversation session ID")
    user_profile: str | None = Field(default=None, description="Optional user background")
    target_city: str | None = Field(default=None, description="Optional target city")
    job_direction: str | None = Field(default=None, description="Optional target job direction")
    resume_text: str | None = Field(default=None, description="Optional resume text")


class StopChatRequest(BaseModel):
    run_id: str = Field(..., description="Agent run ID to stop")


class ClearChatRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session ID to clear")


class ToolCallInfo(BaseModel):
    name: str = Field(..., description="Runtime tool name")
    agent_name: str = Field(..., description="Agent wrapper name")
    display_name: str = Field(..., description="Display name")
    description: str = Field(default="", description="Tool description")
    category: str = Field(default="unknown", description="Tool category")
    requires_network: bool | None = Field(default=None, description="Whether the tool needs network access")
    latency: str | None = Field(default=None, description="Expected latency class")
    evidence_type: str | None = Field(default=None, description="Evidence type")
    status: str | None = Field(default=None, description="Tool status")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent reply")
    session_id: str = Field(..., description="Conversation session ID")
    run_id: str | None = Field(default=None, description="Agent run ID")
    model_provider: str | None = Field(default=None, description="Agent model provider used in this run")
    model_name: str | None = Field(default=None, description="Agent model name used in this run")
    used_subagents: list[str] = Field(default_factory=list, description="Subagents used in this run")
    tool_calls_summary: list[str] = Field(default_factory=list, description="Tool call summary")
    tool_calls: list[ToolCallInfo] = Field(default_factory=list, description="Structured tool call details")
    sources: list[str] = Field(default_factory=list, description="Reference URLs")
    latency_ms: float = Field(..., description="Latency in milliseconds")
    context_compression: dict = Field(default_factory=dict, description="Context compression metadata")
    trace: dict = Field(default_factory=dict, description="Visual trace data")
    error: str | None = Field(default=None, description="Error message when failed")


class ChatHistoryMessage(BaseModel):
    role: str = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    status: str | None = Field(default=None, description="Agent message status")
    meta: dict = Field(default_factory=dict, description="Agent metadata")
    activity: dict = Field(default_factory=dict, description="Agent activity metadata")


class ChatHistoryResponse(BaseModel):
    session_id: str = Field(..., description="Conversation session ID")
    messages: list[ChatHistoryMessage] = Field(default_factory=list, description="Expanded message history")


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    turn_count: int = 0
    message_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionSummary] = Field(default_factory=list)


class ChatModelOption(BaseModel):
    provider: str
    label: str
    model: str
    base_url: str
    configured: bool = False
    default: bool = False


class ChatModelOptionsResponse(BaseModel):
    models: list[ChatModelOption] = Field(default_factory=list)
