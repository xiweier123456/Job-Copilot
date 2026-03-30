from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入的消息")
    session_id: str = Field(default="default", description="会话 ID")
    user_profile: str | None = Field(default=None, description="用户背景信息，可选")
    target_city: str | None = Field(default=None, description="目标城市，可选")
    job_direction: str | None = Field(default=None, description="目标岗位方向，可选")
    resume_text: str | None = Field(default=None, description="简历文本，可选")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent 的回复")
    session_id: str = Field(..., description="会话 ID，原样返回")
    used_subagents: list[str] = Field(default_factory=list, description="本轮实际使用的子 Agent")
    tool_calls_summary: list[str] = Field(default_factory=list, description="本轮识别到的工具调用摘要")
    sources: list[str] = Field(default_factory=list, description="从回复中提取到的参考链接")
    latency_ms: float = Field(..., description="本轮调用耗时，单位毫秒")
    error: str | None = Field(default=None, description="错误信息；成功时为 null")


def _build_agent_message(request: ChatRequest) -> str:
    context_parts: list[str] = []

    if request.user_profile:
        context_parts.append(f"【用户背景】\n{request.user_profile.strip()}")
    if request.target_city:
        context_parts.append(f"【目标城市】\n{request.target_city.strip()}")
    if request.job_direction:
        context_parts.append(f"【目标岗位方向】\n{request.job_direction.strip()}")
    if request.resume_text:
        context_parts.append(f"【简历文本】\n{request.resume_text.strip()}")

    context_parts.append(f"【用户问题】\n{request.message.strip()}")
    return "\n\n".join(context_parts)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    求职问答对话（deepagents / create_deep_agent）。
    Agent 会自主委派 subagent，并通过岗位检索、简历分析、职业路径、面试准备等工具完成回答。
    """
    from app.agents.graph import AgentRunError, run_agent

    try:
        result = await run_agent(_build_agent_message(request), request.session_id)
    except AgentRunError as exc:
        return ChatResponse(
            reply="抱歉，这次对话暂时失败了，请稍后重试。",
            session_id=request.session_id,
            latency_ms=0,
            error=str(exc),
        )

    return ChatResponse(**result)
