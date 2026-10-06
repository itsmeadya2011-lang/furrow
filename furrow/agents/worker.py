from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_EDIT_PATTERN = re.compile(
    r"FILE:\s*(?P<path>\S+)\s*\n\s*```\w*\n(?P<content>.*?)```\s*\n\s*END FILE",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.client.settings.workspace / p
        return p

    def _parse_edits(self, response: str) -> list[tuple[Path, str]]:
        edits: list[tuple[Path, str]] = []
        for m in _EDIT_PATTERN.finditer(response):
            edits.append((self._resolve_path(m.group("path")), m.group("content")))
        return edits

    def _exists(self, path: Path) -> bool:
        return path.exists()

    def _rel(self, path: Path) -> Path:
        try:
            return path.relative_to(self.client.settings.workspace)
        except ValueError:
            return path

    async def run(self) -> str:
        files = self.task.files
        workspace = self.client.settings.workspace

        project_files = self.client.list_files(workspace)
        structure = "\n".join(project_files) if project_files else "(no files found in workspace)"

        file_contents: dict[str, str] = {}
        for f in files:
            path = self._resolve_path(f)
            try:
                file_contents[str(path)] = await self.client.read_file(path)
            except FileNotFoundError:
                file_contents[str(path)] = "<new file - does not exist yet>"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task description: {self.task.description}\n\n"
            f"Files to touch: {', '.join(files) if files else 'none specified'}\n\n"
            f"Project structure (files in workspace):\n{structure}\n\n"
            f"Contents of files to touch:\n"
        )
        for path, content in file_contents.items():
            prompt += f"\n--- {path} ---\n{content}\n"

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        edits = self._parse_edits(response)
        created: list[str] = []
        modified: list[str] = []
        for path, content in edits:
            existed = self._exists(path)
            await self.client.write_file(path, content)
            name = str(self._rel(path))
            if existed:
                modified.append(name)
            else:
                created.append(name)

        if not edits:
            excerpt = response.strip().replace("\n", " ")[:200]
            return f"No file changes were produced. LLM response: {excerpt}"

        parts: list[str] = []
        if modified:
            parts.append(f"Modified {len(modified)} file(s): {', '.join(modified)}")
        if created:
            parts.append(f"Created {len(created)} file(s): {', '.join(created)}")
        return "; ".join(parts)
