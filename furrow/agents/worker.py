from __future__ import annotations

import asyncio
import json
from pathlib import Path
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

    async def write_file(self, path: str | Path, content: str) -> None:
        await self.client.write_file(path, content)

    async def run(self) -> str:
        context = await self._build_file_context()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        if context:
            prompt += f"\n{context}\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        summary = await self._apply_response(response)
        return summary

    async def _build_file_context(self) -> str:
        if not self.task.files:
            return ""

        async def read_one(path_str: str) -> str | None:
            p = Path(path_str)
            try:
                return await self.client.read_file(p)
            except (FileNotFoundError, IOError, OSError):
                return None

        results = await asyncio.gather(*(read_one(f) for f in self.task.files))

        sections: list[str] = []
        for path_str, content in zip(self.task.files, results):
            if content is None:
                continue
            sections.append(f"{path_str}:\n{content}")

        if not sections:
            return ""

        return "Existing file contents:\n\n" + "\n\n".join(sections)

    async def _apply_response(self, response: str) -> str:
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return response

        if not isinstance(data, dict):
            return response

        files = data.get("files")
        if isinstance(files, list):
            for entry in files:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                content = entry.get("content")
                if path is None or content is None:
                    continue
                await self.write_file(path, content)

        summary = data.get("summary")
        if isinstance(summary, str) and summary:
            return summary
        return response