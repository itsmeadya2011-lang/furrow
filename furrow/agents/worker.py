from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Iterator

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

DEFAULT_MAX_TOOL_ROUNDS = 10


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        if not self._file_tools_enabled():
            prompt = self._build_prompt([self._initial_message()])
            return await self.client.complete(prompt, model=self.client.settings.worker_model)

        messages = [self._initial_message()]
        max_rounds = self._max_tool_rounds()
        for _ in range(max_rounds):
            prompt = self._build_prompt(messages)
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            tool_call = self._parse_tool_call(response)
            if tool_call is None:
                return response
            messages.append({"role": "assistant", "content": response})
            try:
                tool_result = await self._execute_tool(tool_call)
            except Exception as e:
                tool_result = f"Error executing tool: {e}"
            messages.append({"role": "tool", "content": tool_result})
        return f"Reached the maximum of {max_rounds} tool rounds without a final answer."

    def _initial_message(self) -> dict[str, str]:
        files = ", ".join(self.task.files) if self.task.files else "any"
        content = (
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n"
            "Start by exploring the repository with list_files to understand the project "
            "structure, then read the relevant files before making any changes."
        )
        return {"role": "user", "content": content}

    def _build_prompt(self, messages: list[dict[str, str]]) -> str:
        parts = [WORKER_PROMPT]
        for message in messages:
            parts.append(f"{message['role']}: {message['content']}")
        return "\n\n".join(parts)

    def _parse_tool_call(self, text: str) -> dict[str, Any] | None:
        chunks: list[str] = []
        for pattern in (r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"):
            chunks.extend(match.group(1) for match in re.finditer(pattern, text, re.DOTALL))
        chunks.append(text)
        for chunk in chunks:
            for raw in self._iter_json_objects(chunk):
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, dict) and "tool" in data:
                    return data
        return None

    def _iter_json_objects(self, text: str) -> Iterator[str]:
        decoder = json.JSONDecoder()
        idx = 0
        while True:
            start = text.find("{", idx)
            if start == -1:
                return
            try:
                _, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                idx = start + 1
                continue
            yield text[start : start + end]
            idx = start + end

    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        tool = tool_call.get("tool")
        if tool == "read_file":
            path = str(tool_call.get("path", ""))
            return await self.client.read_file(path)
        if tool == "write_file":
            path = str(tool_call.get("path", ""))
            content = str(tool_call.get("content", ""))
            await self.client.write_file(path, content)
            return f"Wrote {len(content)} characters to {path}"
        if tool == "list_files":
            directory = str(tool_call.get("directory", "."))
            files = self.client.list_files(directory)
            if not files:
                return f"No files found in {directory}"
            return "\n".join(files)
        return f"Unknown tool: {tool!r}. Available tools: read_file, write_file, list_files"

    def _max_tool_rounds(self) -> int:
        settings = getattr(self.client, "settings", None)
        rounds = getattr(settings, "max_tool_rounds", None)
        return int(rounds) if rounds else DEFAULT_MAX_TOOL_ROUNDS

    def _file_tools_enabled(self) -> bool:
        settings = getattr(self.client, "settings", None)
        return bool(getattr(settings, "use_file_tools", True))
