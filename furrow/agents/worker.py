from __future__ import annotations

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

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.client.settings.workspace / p
        return p

    async def _read_existing(self, path: str) -> str | None:
        p = self._resolve(path)
        try:
            return await self.client.read_file(p)
        except FileNotFoundError:
            return None
        except OSError:
            return None

    @staticmethod
    def _extract_json(text: str) -> dict:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines)
        return json.loads(stripped)

    async def run(self) -> str:
        file_sections: list[str] = []
        for f in self.task.files:
            content = await self._read_existing(f)
            if content is not None:
                file_sections.append(f"<file path=\"{f}\">\n{content}\n</file>")
            else:
                file_sections.append(f"<file path=\"{f}\">\n(NEW FILE)\n</file>")

        files_block = "\n\n".join(file_sections)

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task ID: {self.task.id}\n\n"
            f"Task description:\n{self.task.description}\n\n"
            f"Files to touch (existing contents shown, or marked NEW FILE):\n"
            f"{files_block or '(no files specified)'}\n\n"
            f"Implement the task. Return ONLY valid JSON with this shape:\n"
            f'{{\n  "files": [\n    {{"path": "src/auth.py", "content": "...new content..."}}\n  ],\n  "summary": "What was changed"\n}}\n'
        )

        response = await self.client.complete(
            prompt,
            system="You are a careful coding agent. Return only valid JSON matching the requested shape, with no additional text.",
            model=self.client.settings.worker_model,
        )

        data = self._extract_json(response)
        files = data.get("files") or []
        summary = data.get("summary") or ""

        for entry in files:
            path = entry["path"]
            content = entry["content"]
            await self.client.write_file(self._resolve(path), content)

        return summary if isinstance(summary, str) else str(summary)
