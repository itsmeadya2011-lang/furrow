from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.files_touched: list[str] = []

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"Begin by reading any relevant existing files, then make your changes."
        )
        for _ in range(5):  # max 5 rounds of tool use
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            tool_call = self._parse_tool_call(response)
            if tool_call is None:
                # No tool call — this should be the final JSON summary
                return response
            result = await self._execute_tool(tool_call)
            prompt += f"\n\nTool result:\n{result}\nContinue with the next step."
        return json.dumps({"summary": "Max rounds reached", "files_touched": self.files_touched})

    def _parse_tool_call(self, response: str) -> dict | None:
        """Try to extract a tool call from the LLM response."""
        # Look for JSON blocks that look like tool calls
        match = re.search(r'\{[^{}]*"tool"\s*:\s*"[^"]+"[^{}]*\}', response)
        if not match:
            return None
        try:
            data = json.loads(match.group())
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            return None
        return None

    async def _execute_tool(self, call: dict) -> str:
        tool = call.get("tool")
        args = call.get("args", {})
        if tool == "read_file":
            path = args.get("path", "")
            content = await self.client.read_file(path)
            return f"File {path} contents:\n{content}"
        elif tool == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            await self.client.write_file(path, content)
            if path not in self.files_touched:
                self.files_touched.append(path)
            return f"Wrote {len(content)} bytes to {path}"
        elif tool == "edit_file":
            path = args.get("path", "")
            old_string = args.get("old_string", "")
            new_string = args.get("new_string", "")
            result = await self.client.edit_file(path, old_string, new_string)
            if path not in self.files_touched:
                self.files_touched.append(path)
            return result
        else:
            return f"Unknown tool: {tool}"