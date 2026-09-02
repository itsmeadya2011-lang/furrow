from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import FileOperation, TaskModel, WorkerResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text, count=2)
    return text.strip()


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = self.client.settings.workspace.resolve()

    async def run(self) -> WorkerResult:
        prompt = (
            f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(_strip_fences(response))
        except (json.JSONDecodeError, ValueError):
            return WorkerResult(
                task_id=self.task.id,
                summary=f"Failed to parse LLM response as JSON: {response[:200]}",
                operations=[],
                success=False,
                error="invalid_json",
            )

        summary = str(data.get("summary", ""))
        raw_ops = data.get("operations", []) or []
        if not isinstance(raw_ops, list):
            return WorkerResult(
                task_id=self.task.id,
                summary=summary or "Invalid operations payload",
                operations=[],
                success=False,
                error="invalid_operations",
            )

        results: list[FileOperation] = []
        any_failed = False
        for raw in raw_ops:
            op = FileOperation(
                action=raw.get("action", ""),
                path=raw.get("path", ""),
                content=raw.get("content"),
            )
            try:
                await self._apply(op)
            except Exception as e:
                op.success = False
                op.error = str(e)
                any_failed = True
            results.append(op)

        return WorkerResult(
            task_id=self.task.id,
            summary=summary,
            operations=results,
            success=not any_failed,
            error=None,
        )

    async def _apply(self, op: FileOperation) -> None:
        if op.action not in ("create", "edit", "delete"):
            raise ValueError(f"Unknown action: {op.action!r}")

        rel = op.path.strip()
        if not rel:
            raise ValueError("path is required")
        if rel.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", rel):
            raise ValueError(f"path must be relative: {rel}")

        target = (self.workspace / rel).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError as e:
            raise ValueError(f"path escapes workspace: {rel}") from e

        if op.action == "create":
            if op.content is None:
                raise ValueError("'create' requires content")
            target.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(target, op.content)
            return

        if op.action == "edit":
            if op.content is None:
                raise ValueError("'edit' requires content")
            if not target.exists():
                try:
                    await self.client.read_file(target)
                except FileNotFoundError:
                    pass
                else:
                    raise
                target.parent.mkdir(parents=True, exist_ok=True)
                await self.client.write_file(target, op.content)
                return
            await self.client.read_file(target)
            await self.client.write_file(target, op.content)
            return

        target.unlink()