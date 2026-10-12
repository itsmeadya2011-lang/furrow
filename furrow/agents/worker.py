from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, TaskStatus
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    MAX_ROUNDS = 8

    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        self.task.status = TaskStatus.IN_PROGRESS

        system = f"""{WORKER_PROMPT}

You have access to tools. To use a tool, respond with a JSON block on its own line:
{{"tool": "read_file", "path": "src/foo.py"}}
{{"tool": "write_file", "path": "src/foo.py", "content": "...full file content..."}}
{{"tool": "list_files", "path": "src"}}
{{"tool": "done", "summary": "what you changed"}}

Rules:
- Work only on the assigned task. Do not refactor unrelated code.
- Make minimal, targeted changes.
- Read a file before editing it.
- Do not spawn subagents.
- When finished, respond with the `done` tool only.
"""

        prompt = f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}"

        workspace = Path(self.client.settings.workspace)
        rounds = self.MAX_ROUNDS

        try:
            while rounds > 0:
                rounds -= 1
                response = await self.client.complete(
                    prompt,
                    system=system,
                    model=self.client.settings.worker_model,
                )

                tool_call = self._parse_tool_call(response)
                if tool_call is None:
                    return response.strip()

                tool_name = tool_call.get("tool")
                if tool_name == "read_file":
                    path = workspace / tool_call.get("path", "")
                    content = await self.client.read_file(path)
                    prompt = json.dumps({"tool": "read_file", "path": tool_call.get("path"), "content": content})
                elif tool_name == "write_file":
                    path = workspace / tool_call.get("path", "")
                    content = tool_call.get("content", "")
                    await self.client.write_file(path, content)
                    prompt = json.dumps({"tool": "write_file", "path": tool_call.get("path"), "ok": True})
                elif tool_name == "list_files":
                    path = workspace / tool_call.get("path", "")
                    files = self.client.list_files(path)
                    prompt = json.dumps({"tool": "list_files", "path": tool_call.get("path"), "files": files})
                elif tool_name == "done":
                    summary = tool_call.get("summary", response.strip())
                    self.task.status = TaskStatus.COMPLETED
                    return summary
                else:
                    prompt = json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception:
            self.task.status = TaskStatus.FAILED
            raise
        else:
            self.task.status = TaskStatus.FAILED
            return "Max rounds exceeded; task may be incomplete."

    def _parse_tool_call(self, text: str) -> dict | None:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None