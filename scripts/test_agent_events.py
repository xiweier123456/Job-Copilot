import asyncio
import json
from pathlib import Path

from langgraph.errors import GraphRecursionError

from app.agents.graph import get_agent


TEST_MESSAGE = (
    "帮我了解 2026 年算法工程师校招的真实面经和准备重点，"
    "优先参考牛客网和知乎上的信息，并给我一个简短总结。"
)


def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)[:500]


async def main() -> None:
    agent = await get_agent()
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": TEST_MESSAGE,
            }
        ]
    }
    config = {
        "configurable": {"thread_id": "debug-subagent-routing"},
        "recursion_limit": 30,
    }

    output_path = Path("agent_event_trace.jsonl")
    first_task = None
    subagent_signals = []
    tool_signals = []

    try:
        with output_path.open("w", encoding="utf-8") as f:
            async for event in agent.astream_events(inputs, config=config, version="v2"):
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

                event_name = event.get("event")
                name = event.get("name")
                data = event.get("data", {}) or {}

                if event_name == "on_tool_start" and name == "task" and first_task is None:
                    first_task = data.get("input")

                text = json.dumps(event, ensure_ascii=False, default=str)
                if any(marker in text for marker in ("job-search-agent", "career-agent", "interview-agent", "resume-agent", "general-purpose")):
                    subagent_signals.append(
                        {
                            "event": event_name,
                            "name": name,
                            "data": compact(data),
                        }
                    )

                if any(marker in text for marker in ("tavily-search", "tavily-research", "tavily-extract", "tvly", "read_file", "ls")):
                    tool_signals.append(
                        {
                            "event": event_name,
                            "name": name,
                            "data": compact(data),
                        }
                    )

    except GraphRecursionError as exc:
        print("检测到 agent 递归调用过深。")
        print(str(exc))
    except Exception as exc:
        print(f"运行失败: {exc}")

    print(f"完整事件流已写入: {output_path.resolve()}")

    print("\n首个 task 调用：")
    if first_task is None:
        print("未捕获到 task 调用")
    else:
        print(json.dumps(first_task, ensure_ascii=False, indent=2, default=str))

    print("\n最近的 subagent 信号：")
    for item in subagent_signals[-8:]:
        print("-" * 80)
        print(json.dumps(item, ensure_ascii=False, indent=2))

    print("\n最近的 skill/tool 信号：")
    for item in tool_signals[-12:]:
        print("-" * 80)
        print(json.dumps(item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
