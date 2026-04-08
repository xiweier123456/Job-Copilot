from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import json

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户输入的消息")
    session_id: str = Field(default="default", description="会话 ID")
    user_profile: str | None = Field(default=None, description="用户背景信息，可选")
    target_city: str | None = Field(default=None, description="目标城市，可选")
    job_direction: str | None = Field(default=None, description="目标岗位方向，可选")
    resume_text: str | None = Field(default=None, description="简历文本，可选")


class StopChatRequest(BaseModel):
    run_id: str = Field(..., description="需要停止的运行 ID")


class ClearChatRequest(BaseModel):
    session_id: str = Field(..., description="需要清空的会话 ID")


class ToolCallInfo(BaseModel):
    name: str = Field(..., description="运行时工具名")
    agent_name: str = Field(..., description="agent wrapper 名称")
    display_name: str = Field(..., description="前端展示用中文工具名")
    description: str = Field(default="", description="工具用途说明")
    category: str = Field(default="unknown", description="工具分类")
    requires_network: bool | None = Field(default=None, description="是否需要联网")
    latency: str | None = Field(default=None, description="预计耗时等级")
    evidence_type: str | None = Field(default=None, description="输出证据类型")
    status: str | None = Field(default=None, description="当前工具状态")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent 的回复")
    session_id: str = Field(..., description="会话 ID，原样返回")
    used_subagents: list[str] = Field(default_factory=list, description="本轮实际使用的子 Agent")
    tool_calls_summary: list[str] = Field(default_factory=list, description="本轮识别到的工具调用摘要")
    tool_calls: list[ToolCallInfo] = Field(default_factory=list, description="本轮结构化工具调用信息")
    sources: list[str] = Field(default_factory=list, description="从回复中提取到的参考链接")
    latency_ms: float = Field(..., description="本轮调用耗时，单位毫秒")
    error: str | None = Field(default=None, description="错误信息；成功时为 null")


class ChatHistoryMessage(BaseModel):
    role: str = Field(..., description="消息角色")
    content: str = Field(..., description="消息内容")
    status: str | None = Field(default=None, description="agent 消息状态")
    meta: dict = Field(default_factory=dict, description="agent 元信息")
    activity: dict = Field(default_factory=dict, description="agent 活动信息")


class ChatHistoryResponse(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    messages: list[ChatHistoryMessage] = Field(default_factory=list, description="按时间顺序展开的历史消息")


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_turn_record(request: ChatRequest, result: dict, status: str) -> dict:
    reply = str(result.get("reply") or "").strip()
    error = result.get("error")
    latest_status = "回答已生成"
    error_message = ""

    if status == "stopped":
        latest_status = "已停止生成"
    elif status == "error":
        latest_status = "处理失败"
        error_message = str(error or "agent 运行失败，请稍后重试。")

    return {
        "turn_id": f"{request.session_id}-{int(datetime.now(timezone.utc).timestamp() * 1000)}",
        "session_id": request.session_id,
        "user_message": request.message.strip(),
        "assistant_message": reply,
        "status": status,
        "created_at": _utc_timestamp(),
        "context": {
            "target_city": request.target_city,
            "job_direction": request.job_direction,
            "user_profile": request.user_profile,
            "resume_text": request.resume_text,
        },
        "meta": {
            "latency_ms": result.get("latency_ms"),
            "used_subagents": result.get("used_subagents") or [],
            "tool_calls_summary": result.get("tool_calls_summary") or [],
            "tool_calls": result.get("tool_calls") or [],
            "sources": result.get("sources") or [],
        },
        "activity": {
            "latestStatus": latest_status,
            "todos": [],
            "subagents": result.get("used_subagents") or [],
            "tools": result.get("tool_calls_summary") or [],
            "toolDetails": result.get("tool_calls") or [],
            "errorMessage": error_message,
        },
    }


def _expand_turns_to_messages(session_id: str, turns: list[dict]) -> ChatHistoryResponse:
    messages: list[ChatHistoryMessage] = []

    for turn in turns:
        user_message = str(turn.get("user_message") or "").strip()
        if user_message:
            messages.append(ChatHistoryMessage(role="user", content=user_message))

        assistant_message = str(turn.get("assistant_message") or "").strip()
        messages.append(
            ChatHistoryMessage(
                role="agent",
                content=assistant_message or ("本次回答已停止。" if turn.get("status") == "stopped" else ""),
                status=str(turn.get("status") or "done"),
                meta=turn.get("meta") or {},
                activity=turn.get("activity") or {},
            )
        )

    return ChatHistoryResponse(session_id=session_id, messages=messages)


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


def _format_sse(event: dict) -> str:
    event_type = event.get("type", "message")
    data = json.dumps(event, ensure_ascii=False)
    return f"event: {event_type}\ndata: {data}\n\n"


def _empty_chat_result(session_id: str, error: str | None = None) -> dict:
    return {
        "reply": "",
        "session_id": session_id,
        "used_subagents": [],
        "tool_calls_summary": [],
        "tool_calls": [],
        "sources": [],
        "latency_ms": 0,
        "error": error,
    }


async def _stream_chat_request(http_request: Request, request: ChatRequest) -> StreamingResponse:
    from app.agents.graph import AgentRunError, cancel_run, create_run, stream_agent_events
    from app.rag.chat_memory_store import save_chat_memory
    from app.services.chat_history_service import save_chat_turn

    message = _build_agent_message(request)
    run = create_run(request.session_id)

    async def event_generator():
        final_payload: dict | None = None
        final_status = "stopped"

        try:
            async for event in stream_agent_events(message, request.session_id, run.run_id):
                event_type = event.get("type")
                payload = event.get("payload") or {}

                if event_type == "final":
                    final_payload = payload
                    final_status = "error" if payload.get("error") else "done"
                elif event_type == "error":
                    final_payload = _empty_chat_result(
                        request.session_id,
                        error=payload.get("message") or "agent 运行失败，请稍后重试。",
                    )
                    final_status = "error"
                elif event_type == "stopped":
                    final_payload = _empty_chat_result(request.session_id)
                    final_status = "stopped"

                if await http_request.is_disconnected():
                    cancel_run(run.run_id)
                    final_status = "stopped"
                    if final_payload is None:
                        final_payload = _empty_chat_result(request.session_id)
                    break

                yield _format_sse(event)
        except AgentRunError:
            final_payload = _empty_chat_result(request.session_id, error="agent 运行失败，请稍后重试。")
            final_status = "error"
        finally:
            if final_payload is not None:
                turn = _build_turn_record(request, final_payload, status=final_status)
                await save_chat_turn(turn)
                await save_chat_memory(turn)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _build_chat_request_from_upload(
    *,
    resume_file: UploadFile,
    message: str,
    session_id: str = "default",
    user_profile: str = "",
    target_city: str = "",
    job_direction: str = "",
) -> ChatRequest:
    from app.services.document_service import extract_text_from_pdf_upload

    resume_text = await extract_text_from_pdf_upload(resume_file)
    return ChatRequest(
        message=message,
        session_id=session_id,
        user_profile=user_profile or None,
        target_city=target_city or None,
        job_direction=job_direction or None,
        resume_text=resume_text,
    )


@router.post("/stream")
async def chat_stream(http_request: Request, request: ChatRequest):
    return await _stream_chat_request(http_request, request)


@router.post("/stream/upload")
async def chat_stream_upload(
    http_request: Request,
    resume_file: UploadFile = File(..., description="上传的 PDF 简历"),
    message: str = Form(..., description="用户输入的消息"),
    session_id: str = Form(default="default", description="会话 ID"),
    user_profile: str = Form(default="", description="用户背景信息，可选"),
    target_city: str = Form(default="", description="目标城市，可选"),
    job_direction: str = Form(default="", description="目标岗位方向，可选"),
):
    request = await _build_chat_request_from_upload(
        resume_file=resume_file,
        message=message,
        session_id=session_id,
        user_profile=user_profile,
        target_city=target_city,
        job_direction=job_direction,
    )
    return await _stream_chat_request(http_request, request)


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str = "default"):
    from app.services.chat_history_service import get_chat_history as load_chat_history

    turns = await load_chat_history(session_id)
    return _expand_turns_to_messages(session_id, turns)


@router.post("/clear")
async def clear_chat(request: ClearChatRequest):
    from app.agents.graph import clear_session_runtime_state
    from app.rag.chat_memory_store import clear_chat_memory
    from app.services.chat_history_service import clear_chat_history

    await clear_chat_history(request.session_id)
    await clear_chat_memory(request.session_id)
    await clear_session_runtime_state(request.session_id)
    return {"ok": True, "session_id": request.session_id}


@router.post("/stop")
async def stop_chat(request: StopChatRequest):
    from app.agents.graph import cancel_run

    return {"ok": True, "stopped": cancel_run(request.run_id)}
