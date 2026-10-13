from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from .logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.logger = get_logger(__name__)

    async def run(self) -> str:
        self.logger.info("Starting worker task", task_id=self.task.id, description=self.task.description)
        messages = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}"},
        ]

        max_iterations = 10
        for _ in range(max_iterations):
            response = await self.client.chat(messages, model=self.client.settings.worker_model)
            messages.append({"role": "assistant", "content": response})

            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                return response

            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                args = tool_call["args"]
                self.logger.debug("Executing tool", tool_name=tool_name, args=args)
                try:
                    if tool_name == "read_file":
                        result = await self.client.read_file(args["path"])
                    elif tool_name == "write_file":
                        await self.client.write_file(args["path"], args["content"])
                        result = f"Successfully wrote to {args['path']}"
                    elif tool_name == "list_files":
                        files = self.client.list_files(args["directory"])
                        result = "\n".join(files) if files else "(no files)"
                    else:
                        result = f"Unknown tool: {tool_name}"
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"
                self.logger.debug("Tool result", tool_name=tool_name, result=result)
                messages.append({"role": "user", "content": f"TOOL RESULT ({tool_name}):\n{result}"})

        return "Max iterations reached without completion"

    def _parse_tool_calls(self, text: str) -> list[dict]:
        pattern = r"TOOL:\s*(\w+)\s*(\{.*?\})"
        calls = []
        for match in re.finditer(pattern, text):
            name = match.group(1)
            try:
                args = json.loads(match.group(2))
                calls.append({"name": name, "args": args})
            except json.JSONDecodeError:
                continue
        return calls