from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_ROUNDS = 15


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self._rounds = 0

    async def run(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n"
        )
        transcript: list[str] = []
        while self._rounds < MAX_ROUNDS:
            self._rounds += 1
            response = await self.client.complete(
                prompt, model=self.client.settings.worker_model
            )
            tool_calls = _extract_tool_calls(response)
            if not tool_calls:
                transcript.append(response)
                return response
            transcript.append(response)
            prompt += f"\n\n[Round {self._rounds} response]\n{response}\n"
            for call in tool_calls:
                result = await _execute_tool(self.client, call)
                prompt += f"[Tool {call['name']} result]\n{result}\n"
        return "\n".join(transcript) or "Max rounds reached."


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:  # noqa: F821
    calls: list[dict[str, Any]] = []  # noqa: F821
    for match in re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            calls.append(obj)
    return calls


async def _execute_tool(client: LLMClient, call: dict[str, Any]) -> str:
    name = call.get("tool")
    args = call.get("args", {})
    try:
        if name == "read_file":
            return await client.read_file(args["path"])
        if name == "write_file":
            await client.write_file(args["path"], args["content"])
            return f"Wrote {args['path']}"
        if name == "list_files":
            files = client.list_files(args.get("directory", "."))
            return "\n".join(files) or "(no files)"
        return f"Unknown tool: {name}"
    except Exception as e:  # noqa: BLE001
        return f"Tool error: {e}"