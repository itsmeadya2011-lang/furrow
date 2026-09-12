from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


# Tool definitions shared across providers.
TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file from the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the workspace.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write or overwrite a file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file within the workspace.",
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write to the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": "Execute a shell command in the workspace directory. Use for tests, lint, type checks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
        },
    },
]


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = self.client.settings.workspace
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Workspace: {workspace}\n"
        )
        messages = [{"role": "user", "content": prompt}]

        for _ in range(20):
            response = await self.client.complete_with_tools(
                prompt=messages[-1]["content"],
                system=WORKER_PROMPT,
                tools=TOOL_DEFINITIONS,
                model=self.client.settings.worker_model,
            )
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                return response.get("text", "").strip() or "Task completed."

            tool_results = []
            for call in tool_calls:
                name = call["name"]
                inp = call["input"]
                try:
                    if name == "read_file":
                        result = await self.client.read_file(workspace / inp["path"])
                    elif name == "write_file":
                        await self.client.write_file(workspace / inp["path"], inp["content"])
                        result = f"Wrote {len(inp['content'])} bytes to {inp['path']}"
                    elif name == "run_command":
                        result = await self.client.run_command(inp["command"])
                    else:
                        result = f"Unknown tool: {name}"
                except Exception as e:
                    result = f"Error: {e}"
                tool_results.append(f"Tool {name} result:\n{result}")

            messages.append({"role": "assistant", "content": response.get("text", "")})
            messages.append(
                {"role": "user", "content": "\n\n".join(tool_results)}
            )
            prompt = "\n\n".join(tool_results)

        return "Task completed (max iterations reached)."