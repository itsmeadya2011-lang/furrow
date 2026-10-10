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

    async def run(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {files}\n"
        raw = await self.client.complete(prompt, model=self.client.settings.worker_model)

        data = None
        try:
            from furrow.llm import extract_json
            data = json.loads(extract_json(raw))
        except Exception:  # noqa: BLE001 - intentional: JSON parse fallback
            try:
                data = json.loads(raw)
            except Exception:  # noqa: BLE001 - intentional: JSON parse fallback
                return raw

        if not isinstance(data, dict):
            return raw

        summary = data.get("summary", "") or raw
        if not isinstance(summary, str):
            summary = str(summary)
        edits = data.get("edits") or []
        if not isinstance(edits, list):
            edits = []

        workspace = Path(self.client.settings.workspace).resolve()
        written = 0
        rejected: list[str] = []
        failed: list[str] = []

        for edit in edits:
            if not isinstance(edit, dict):
                failed.append("invalid edit entry")
                continue
            path = edit.get("path")
            content = edit.get("content", "")
            if not isinstance(path, str) or not path:
                failed.append("missing path")
                continue
            if not isinstance(content, str):
                content = str(content)
            candidate = Path(path)
            target = candidate if candidate.is_absolute() else workspace / candidate
            resolved = target.resolve()
            if not resolved.is_relative_to(workspace):
                rejected.append(path)
                continue
            try:
                await self.client.write_file(resolved, content)
                written += 1
            except Exception as exc:  # noqa: BLE001 - intentional: per-file write errors collected
                failed.append(f"{path}: {exc}")

        parts: list[str] = []
        if written:
            parts.append(f"{written} files written")
        if failed:
            parts.append(f"{len(failed)} failed")
        if rejected:
            parts.append(f"{len(rejected)} rejected: {', '.join(rejected)}")
        if parts:
            suffix = ", ".join(parts)
            summary = f"{summary} ({suffix})" if summary else suffix
        return summary
