from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.tools import (
    ANTHROPIC_TOOLS,
    OPENAI_TOOLS,
    edit_file,
    grep_files,
    list_files,
    read_file,
    run_command,
    write_file,
)

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )

        # Choose tool schema format based on provider
        if self.client.settings.provider.value == "anthropic":
            tools = ANTHROPIC_TOOLS
        else:
            tools = OPENAI_TOOLS

        async def tool_executor(tool_name: str, tool_input: dict) -> str:
            try:
                if tool_name == "read_file":
                    return await read_file(tool_input["path"])
                elif tool_name == "write_file":
                    await write_file(tool_input["path"], tool_input["content"])
                    return f"Wrote {tool_input['path']}"
                elif tool_name == "edit_file":
                    await edit_file(tool_input["path"], tool_input["old_text"], tool_input["new_text"])
                    return f"Edited {tool_input['path']}"
                elif tool_name == "list_files":
                    files = await list_files(tool_input.get("directory", "."))
                    return "\n".join(files) if files else "No files found"
                elif tool_name == "grep_files":
                    return await grep_files(tool_input["pattern"], tool_input.get("directory", "."))
                elif tool_name == "run_command":
                    return await run_command(tool_input["command"])
                else:
                    return f"Unknown tool: {tool_name}"
            except Exception as e:
                return f"Error: {e}"

        return await self.client.complete_with_tools(
            prompt=prompt,
            system=WORKER_PROMPT,
            model=self.client.settings.worker_model,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_rounds=15,
        )
