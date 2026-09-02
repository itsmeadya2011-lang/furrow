from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.core.tools import ToolRegistry
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        registry = ToolRegistry()
        prompt = (
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        conversation = f"Human: {prompt}"
        tool_pattern = re.compile(r"^TOOL:\s*(\w+)\((.*)\)\s*$", re.DOTALL)

        for _ in range(5):
            response = await self.client.complete(
                conversation,
                system=WORKER_PROMPT,
                model=self.client.settings.worker_model,
            )
            response = response.strip()
            match = tool_pattern.match(response)
            if not match:
                return response

            tool_name = match.group(1)
            args = self._parse_args(match.group(2).strip())
            result = await registry.execute(tool_name, *args)
            conversation += f"\nAssistant: {response}\nTool result: {result}\n"

        return response

    @staticmethod
    def _parse_args(args_str: str) -> list[str]:
        if not args_str:
            return []
        parts: list[str] = []
        current: list[str] = []
        in_quotes = False
        quote_char = ""
        for char in args_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
                current.append(char)
            elif char == quote_char and in_quotes:
                in_quotes = False
                current.append(char)
                quote_char = ""
            elif char == "," and not in_quotes:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        parts.append("".join(current).strip())
        result = []
        for part in parts:
            part = part.strip()
            if len(part) >= 2 and part[0] in ('"', "'") and part[-1] == part[0]:
                part = part[1:-1]
            result.append(part)
        return result

