from __future__ import annotations

import asyncio
import re
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

    async def run(self) -> str:
        context_files: dict[str, str] = {}
        for rel_path in self.task.files:
            full_path = Path(self.client.settings.workspace) / rel_path
            if full_path.exists() and full_path.is_file():
                try:
                    context_files[rel_path] = await self.client.read_file(full_path)
                except OSError:
                    continue

        file_context = "\n\n".join(
            f"--- {path} ---\n{content}" for path, content in context_files.items()
        )

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"Current file contents:\n{file_context}\n\n"
            "Return your changes in this exact format for each modified file:\n"
            "--- path/to/file.py ---\n"
            "<full new file content>\n"
            "--- end ---\n"
            "Then provide a brief summary below."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        written = await self._apply_changes(response)
        summary = self._extract_summary(response)
        parts = [summary] if summary else []
        parts.extend(written)
        return "\n".join(parts) if parts else summary or "No changes applied."

    async def _apply_changes(self, response: str) -> list[str]:
        pattern = re.compile(
            r"^---\s*(?P<path>[^-\n]+?)\s*---\s*\n(?P<content>.*?)\n---\s*end\s*---",
            re.DOTALL | re.MULTILINE,
        )
        written: list[str] = []
        for match in pattern.finditer(response):
            rel_path = match.group("path").strip()
            content = match.group("content")
            full_path = Path(self.client.settings.workspace) / rel_path
            try:
                await self.client.write_file(full_path, content)
                written.append(f"Wrote {rel_path}")
            except OSError as e:
                written.append(f"Failed to write {rel_path}: {e}")
        return written

    def _extract_summary(self, response: str) -> str:
        match = re.search(r"---\s*end\s*---\s*\n(.+)$", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = [line for line in response.strip().splitlines() if not line.startswith("---")]
        return "\n".join(lines).strip()
