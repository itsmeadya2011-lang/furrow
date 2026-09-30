from __future__ import annotations

import re

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

_FILE_BLOCK = re.compile(
    r"===FURROW_FILE:\s*(\S+)\s*===\n(.*?)\n===FURROW_END===",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())

    @staticmethod
    def parse_files(response: str) -> list[tuple[str, str]]:
        return [(m.group(1).strip(), m.group(2)) for m in _FILE_BLOCK.finditer(response)]

    async def _read_context(self) -> str:
        files = self.task.files or []
        if not files:
            return ""
        chunks: list[str] = []
        for f in files:
            try:
                content = await self.client.read_file(f)
                chunks.append(f"--- Existing file: {f} ---\n{content}\n--- End {f} ---")
            except Exception as e:  # noqa: BLE001
                chunks.append(f"--- Existing file: {f} (could not read: {e}) ---")
        return "\n".join(chunks)

    async def run(self) -> str:
        context = await self._read_context()
        files_hint = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_hint}\n"
            f"{context}\n"
            f"Return changes as ===FURROW_FILE: <path>=== ... ===FURROW_END=== blocks."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        writes = self.parse_files(response)
        written: list[str] = []
        for path, content in writes:
            await self.client.write_file(path, content)
            written.append(path)
        parts: list[str] = []
        if written:
            parts.append(f"Wrote {len(written)} file(s): {', '.join(written)}")
        else:
            parts.append("No file changes were produced.")
            parts.append(f"LLM response: {response[:500]}")
        return "\n".join(parts)
