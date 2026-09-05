from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    pass


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or self.client.settings.workspace

    async def run(self) -> str:
        context = await self._build_context()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"{context}"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        written = await self._apply_changes(response)
        summary = f"Implemented task '{self.task.id}'. Modified files: {', '.join(written) if written else 'none'}. LLM notes: {response[:200]}..."
        return summary

    async def _build_context(self) -> str:
        if not self.task.files:
            return ""
        parts = ["Current file contents:"]
        for rel_path in self.task.files:
            full_path = self.workspace / rel_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    parts.append(f"\n--- {rel_path} ---\n{content}\n")
                except Exception:
                    parts.append(f"\n--- {rel_path} ---\n[Could not read file]\n")
            else:
                parts.append(f"\n--- {rel_path} ---\n[File does not exist yet]\n")
        return "\n".join(parts)

    async def _apply_changes(self, response: str) -> list[str]:
        written: list[str] = []
        pattern = re.compile(r"```([^\n]+)\n(.*?)```", re.DOTALL)
        for match in pattern.finditer(response):
            filename = match.group(1).strip()
            content = match.group(2)
            # Remove trailing newline added by the code block formatting
            if content.endswith("\n"):
                content = content[:-1]
            target = self.workspace / filename
            try:
                await self.client.write_file(target, content)
                written.append(filename)
            except Exception as e:
                # Log but continue applying other files
                pass
        return written
