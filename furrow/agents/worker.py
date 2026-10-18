from __future__ import annotations

import re
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
        # Build context: read existing files the task touches
        context = await self._build_context()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Dependencies: {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}\n\n"
            f"Current project state:\n{context}\n\n"
            f"Return your changes as fenced code blocks with file paths:\n"
            f"```filename\n<content>\n```\n"
            f"Only include files you actually changed."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        await self._apply_changes(response)
        return f"Completed: {self.task.description}"

    async def _build_context(self) -> str:
        """Read existing files in the workspace for context."""
        if not self.task.files:
            # Provide a high-level listing of the workspace
            files = self.client.list_files(self.client.settings.workspace)
            if not files:
                return "(empty workspace)"
            return "Workspace files:\n" + "\n".join(f"  {f}" for f in files[:30])

        lines = []
        for filepath in self.task.files:
            full_path = self.client.settings.workspace / filepath
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    lines.append(f"--- {filepath} (existing) ---\n{content}\n")
                except Exception as e:
                    lines.append(f"--- {filepath} --- (error reading: {e})\n")
            else:
                lines.append(f"--- {filepath} --- (new file)\n")
        return "\n".join(lines) if lines else "(no context available)"

    async def _apply_changes(self, response: str) -> None:
        """Parse fenced code blocks from the LLM response and write them to disk."""
        pattern = re.compile(r"```(\S+)?\n(.*?)```", re.DOTALL)
        matches = pattern.findall(response)
        if not matches:
            return

        written = 0
        for filename, content in matches:
            filename = filename.strip()
            if not filename or filename in ("text", "python", "json", "bash", "sh", "yaml", "yml", "toml"):
                # Skip generic language tags — they aren't file paths
                continue
            # Strip leading/trailing path separators
            filename = filename.strip("/")
            if not filename or "/" not in filename and "." not in filename:
                continue
            full_path = self.client.settings.workspace / filename
            await self.client.write_file(full_path, content.rstrip("\n") + "\n")
            written += 1

        if written == 0:
            # No valid file blocks found; log the raw response for debugging
            pass