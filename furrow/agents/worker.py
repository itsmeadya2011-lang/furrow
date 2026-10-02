from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_MAX_CONTEXT_CHARS = 16000


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    def _safe_target(self, path: str, workspace: Path) -> Path:
        target = (workspace / path).resolve()
        root = workspace.resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Refusing to write outside workspace: {path}")
        return target

    async def _load_context(self, workspace: Path) -> str:
        if not self.task.files:
            return "any file may be created/modified"
        parts: list[str] = []
        total = 0
        for rel in self.task.files:
            source = workspace / rel
            if not source.is_file():
                continue
            try:
                content = await self.client.read_file(source)
            except OSError:
                continue
            block = f"### {rel}\n{content}\n"
            if total + len(block) > _MAX_CONTEXT_CHARS:
                remaining = _MAX_CONTEXT_CHARS - total
                if remaining > 0:
                    parts.append(block[:remaining])
                parts.append("\n[truncated context to fit size limit]\n")
                break
            parts.append(block)
            total += len(block)
        return "".join(parts)

    async def run(self) -> str:
        workspace = self.client.settings.workspace
        context = await self._load_context(workspace)
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n\n"
            f"Existing file context:\n{context}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        text = response.strip()
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse worker output:\n{response}") from exc

        files = (data.get("files") if isinstance(data, dict) else None) or []
        written: list[str] = []
        for entry in files:
            path = entry["path"]
            content = entry["content"]
            target = self._safe_target(path, workspace)
            await self.client.write_file(target, content)
            written.append(path)

        summary = data.get("summary")
        if not summary:
            summary = f"Wrote: {', '.join(written) or 'no files'}"
        return summary
