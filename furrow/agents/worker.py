from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT, WORKER_TOOLS
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_iterations = 10

    async def run(self) -> str:
        system_prompt = WORKER_PROMPT
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"}
        ]

        for iteration in range(self.max_iterations):
            response = await self.client.complete_with_tools(
                messages=messages,
                system=system_prompt,
                tools=WORKER_TOOLS,
                model=self.client.settings.worker_model,
            )

            if response.get("tool_calls"):
                # Execute tool calls
                for tool_call in response["tool_calls"]:
                    tool_result = await self._execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result,
                    })
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": response.get("content", ""),
                    "tool_calls": response["tool_calls"],
                })
            else:
                # Final response - no more tool calls
                return response.get("content", "Task completed (no output)")

        return "Max iterations reached. Task may be incomplete."

    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        try:
            if name == "read_file":
                content = await self.client.read_file(args["path"])
                return content
            elif name == "write_file":
                await self.client.write_file(args["path"], args["content"])
                return f"Successfully wrote to {args['path']}"
            elif name == "list_files":
                files = self.client.list_files(args["directory"])
                return "\n".join(files) if files else "No files found"
            elif name == "run_command":
                import asyncio
                cmd = args["command"]
                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                return f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error executing {name}: {e}"
