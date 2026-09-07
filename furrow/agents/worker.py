from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TOOL_SCHEMA, WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    MAX_ROUNDS = 15

    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    def _extract_tool_actions(self, response: str) -> list[dict]:
        actions: list[dict] = []
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped.startswith("{") or not stripped.endswith("}"):
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and "action" in data:
                actions.append(data)
        return actions

    async def _execute_tool(self, action: str, params: dict) -> str:
        if action == "read_file":
            path = params.get("path", "")
            content = await self.client.read_file(path)
            return f"--- {path} ---\n{content}"
        elif action == "write_file":
            path = params.get("path", "")
            content = params.get("content", "")
            await self.client.write_file(path, content)
            return f"Wrote {len(content)} bytes to {path}"
        elif action == "list_files":
            directory = params.get("path", ".")
            files = self.client.list_files(directory)
            return "\n".join(files) if files else "(empty directory)"
        elif action == "run_shell":
            command = params.get("command", "")
            returncode, stdout, stderr = await self.client.run_shell(command)
            parts = [f"exit code: {returncode}"]
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            return "\n".join(parts)
        else:
            return f"Unknown action: {action}"

    def _format_conversation(self, messages: list[tuple[str, str]]) -> str:
        blocks: list[str] = []
        for role, content in messages:
            if role == "user":
                blocks.append(f"Human: {content}")
            elif role == "assistant":
                blocks.append(f"Assistant: {content}")
            elif role == "system":
                blocks.append(f"System: {content}")
        return "\n\n".join(blocks)

    async def run(self) -> str:
        system_prompt = f"{WORKER_PROMPT}\n\n{TOOL_SCHEMA}\n\nInstructions: Emit exactly one JSON tool action per line when you need to use a tool. When the task is fully complete, reply with a concise plain-text summary and no JSON blocks."
        task_prompt = f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}"
        messages: list[tuple[str, str]] = [("user", task_prompt)]

        for _ in range(self.MAX_ROUNDS):
            prompt = self._format_conversation(messages)
            response = await self.client.complete(prompt, system=system_prompt, model=self.client.settings.worker_model)
            messages.append(("assistant", response))
            actions = self._extract_tool_actions(response)
            if not actions:
                return response
            for action in actions:
                result = await self._execute_tool(action.get("action", ""), action)
                messages.append(("user", f"Tool result for {action.get('action', '')}: {result}"))

        final_messages = [m for m in messages if m[0] == "assistant"]
        return final_messages[-1][1] if final_messages else "No response"
