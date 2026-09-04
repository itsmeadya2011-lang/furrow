from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


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
        self.workspace = workspace or Path.cwd()

    async def run(self) -> str:
        context = await self._build_context()
        prompt = f"""{WORKER_PROMPT}

{context}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Output format:
For each file you create or modify, use:

FILE: <relative-path>
```<language>
<file-content>
```

Return only the files you changed. Keep changes minimal and focused."""
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        written = self._write_files_from_response(response)
        summary = f"Worker completed task {self.task.id}. Wrote {len(written)} file(s)."
        if written:
            summary += f" Files: {', '.join(written)}"
        return summary

    async def _build_context(self) -> str:
        parts = []
        for rel_path in self.task.files:
            full_path = self.workspace / rel_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(full_path)
                    parts.append(f"--- {rel_path} ---\n{content}\n")
                except Exception:
                    pass
        if not parts:
            all_files = self.client.list_files(self.workspace)
            if all_files:
                parts.append(f"Workspace files: {', '.join(all_files[:20])}")
        return "\n".join(parts) if parts else "No existing file context available."

    def _write_files_from_response(self, response: str) -> list[str]:
        pattern = re.compile(r"FILE:\s*(.+?)\s*```.*?\n(.*?)```", re.DOTALL)
        written: list[str] = []
        for match in pattern.finditer(response):
            rel_path = match.group(1).strip()
            content = match.group(2)
            target = self.workspace / rel_path
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written.append(rel_path)
            except Exception:
                continue
        return written
