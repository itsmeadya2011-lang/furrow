from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


FILE_BLOCK_RE = re.compile(r"FILE:\s*(.+?)\n```(\w+)?\n(.*?)```", re.DOTALL)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = Path(self.client.settings.workspace)
        context_parts: list[str] = []
        context_parts.append(f"{WORKER_PROMPT}")
        context_parts.append(f"\nTask: {self.task.description}")
        files_to_touch = self.task.files if self.task.files else []
        context_parts.append(f"\nFiles to touch: {', '.join(files_to_touch) if files_to_touch else 'any'}")

        # Read existing files in the workspace for context
        try:
            all_files = self.client.list_files(workspace)
        except Exception:
            all_files = []

        # Prioritize files mentioned in task
        files_to_read = []
        for ft in files_to_touch:
            if ft in all_files or Path(workspace, ft).exists():
                files_to_read.append(ft)

        # Read up to a reasonable number of additional files for context
        if not files_to_read:
            files_to_read = all_files[:20]

        if files_to_read:
            context_parts.append("\n\nExisting file contents:")
            for fname in files_to_read[:20]:
                try:
                    fpath = Path(workspace, fname)
                    content = await self.client.read_file(fpath)
                    context_parts.append(f"\n--- {fname} ---\n{content}")
                except Exception:
                    pass

        prompt = "\n".join(context_parts)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse FILE blocks
        written_files: list[str] = []
        for match in FILE_BLOCK_RE.finditer(response):
            rel_path = match.group(1).strip()
            file_content = match.group(3)
            try:
                out_path = Path(workspace, rel_path)
                await self.client.write_file(out_path, file_content)
                written_files.append(rel_path)
            except Exception:
                pass

        if written_files:
            return f"Wrote files: {', '.join(written_files)}\n\n{response}"
        return response
