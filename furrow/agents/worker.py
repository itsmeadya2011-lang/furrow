from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_TOOL_DESCRIPTIONS = """You have access to these tools. Respond with a single JSON object (no markdown fences, no extra text):
- {"tool": "list_files", "path": "."}
- {"tool": "read_file", "path": "src/main.py"}
- {"tool": "write_file", "path": "src/main.py", "content": "new file content"}
- {"tool": "done", "summary": "What you changed and any issues"}

Always respond with a single JSON object. Do not include markdown fences or extra text."""


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n{_TOOL_DESCRIPTIONS}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        messages = [{"role": "user", "content": prompt}]
        max_rounds = 10

        for _ in range(max_rounds):
            response = await self.client.complete(
                self._render_messages(messages),
                model=self.client.settings.worker_model,
            )
            tool_call = self._parse_tool_call(response)
            if tool_call is None:
                return response.strip()
            if tool_call.get("tool") == "done":
                return tool_call.get("summary", response)
            result = await self._execute_tool(tool_call)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Tool result:\n{result}\n"})

        return "Worker reached max rounds without completing."

    def _render_messages(self, messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m["role"].capitalize()
            parts.append(f"### {role}\n{m['content']}")
        return "\n\n".join(parts)

    def _parse_tool_call(self, response: str) -> dict | None:
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r"```(?:json)?\n?", "", text)
            text = text.strip()
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    async def _execute_tool(self, call: dict) -> str:
        tool = call.get("tool")
        if tool == "list_files":
            path = call.get("path", ".")
            files = self.client.list_files(path)
            return json.dumps(files, indent=2)
        elif tool == "read_file":
            path = call.get("path", "")
            try:
                content = await self.client.read_file(path)
                return content
            except Exception as e:
                return f"Error reading {path}: {e}"
        elif tool == "write_file":
            path = call.get("path", "")
            content = call.get("content", "")
            try:
                await self.client.write_file(path, content)
                return f"Written {len(content)} bytes to {path}"
            except Exception as e:
                return f"Error writing {path}: {e}"
        else:
            return f"Unknown tool: {tool}"
