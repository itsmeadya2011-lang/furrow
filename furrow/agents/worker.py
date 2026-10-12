from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, WORKER_TOOLS_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_tool_calls(text: str) -> tuple[list[dict], str]:
    """Extract tool_call JSON blocks from LLM text. Returns (calls, cleaned_text)."""
    calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    cleaned = _TOOL_CALL_RE.sub("", text).strip()
    return calls, cleaned


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self._max_rounds = 3

    async def run(self) -> str:
        prompt = (
            f"{WORKER_TOOLS_PROMPT}\n\n"
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        conversation: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        summary = ""

        for _ in range(self._max_rounds):
            response = await self.client.complete(
                prompt=self._format_messages(conversation),
                model=self.client.settings.worker_model,
            )
            tool_calls, cleaned = _extract_tool_calls(response)
            if not tool_calls:
                summary = cleaned or response
                break

            conversation.append({"role": "assistant", "content": response})
            for call in tool_calls:
                result = await self._execute_tool(call)
                conversation.append({"role": "user", "content": f"<tool_result>{result}</tool_result>"})
        else:
            summary = "Max tool rounds reached."

        return summary or "Task complete."

    def _format_messages(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            parts.append(f"[{msg['role']}]\n{msg['content']}")
        return "\n\n".join(parts)

    async def _execute_tool(self, call: dict) -> str:
        name = call.get("name", "")
        inputs = call.get("input", {})
        try:
            if name == "read_file":
                content = await self.client.read_file(inputs["path"])
                return f"File {inputs['path']} ({len(content)} chars):\n{content[:2000]}"
            elif name == "write_file":
                await self.client.write_file(inputs["path"], inputs["content"])
                return f"Wrote {len(inputs['content'])} chars to {inputs['path']}"
            elif name == "list_files":
                files = self.client.list_files(inputs.get("directory", "."))
                return f"Files: {', '.join(files[:50])}"
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Tool {name} failed: {e}"