from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import FileOperation, TaskModel, WorkerResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class WorkerAgent:
    """Agent that implements a single task by writing/patching files.

    The agent reads existing context files, asks the LLM for a structured JSON
    response describing file operations, then applies those operations to disk.
    """

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

    async def _read_context(self) -> str:
        """Read contents of files listed in the task for context.

        Returns a formatted string with file paths and contents, or empty
        string if no files are listed.
        """
        if not self.task.files:
            return ""
        blocks: list[str] = []
        for rel in self.task.files:
            path = Path(rel)
            if not path.is_absolute():
                path = self.workspace / path
            try:
                text = await self.client.read_file(path)
            except (FileNotFoundError, IsADirectoryError, PermissionError, OSError) as e:
                await log.awarn("worker.read_context.missing", path=str(path), error=str(e))
                continue
            blocks.append(f"Path: {rel}\n{text}")
        return "\n\n".join(blocks)

    def _resolve_path(self, rel: str) -> Path:
        """Resolve a possibly-relative path against the workspace root."""
        p = Path(rel)
        if not p.is_absolute():
            p = self.workspace / p
        return p.resolve()

    async def _apply_operation(self, op: FileOperation) -> str | None:
        """Apply a single FileOperation to disk.

        Returns an error string on failure, or None on success.
        """
        target = self._resolve_path(op.path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)

            if op.operation == "edit":
                if not target.exists():
                    return f"Cannot edit non-existent file: {op.path}"
                original = await self.client.read_file(target)
                if op.old_str is None or op.old_str not in original:
                    return f"old_str not found in {op.path}"
                updated = original.replace(op.old_str, op.new_str, 1)
                await self.client.write_file(target, updated)
            else:
                if op.content is None:
                    return f"Write operation missing content for {op.path}"
                await self.client.write_file(target, op.content)

            await log.ainfo("worker.apply_operation", path=str(target), operation=op.operation)
            return None
        except Exception as e:
            await log.aerror("worker.apply_failed", path=op.path, error=str(e))
            return str(e)

    def _parse_response(self, response: str) -> WorkerResult:
        """Parse the LLM response into a WorkerResult.

        Tolerates markdown fences. Falls back to creating a single write
        operation if the response looks like file content, or returns the raw
        text as the summary.
        """
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.removeprefix("```json").removeprefix("```").strip()
            if raw.endswith("```"):
                raw = raw.removesuffix("```").strip()

        try:
            data = json.loads(raw)
            return WorkerResult(**data)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        if self.task.files and (
            "\n" in raw
            or raw.endswith((".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml"))
        ):
            target = self.task.files[0]
            return WorkerResult(
                summary=f"Applied raw response as write to {target}",
                operations=[FileOperation(path=target, operation="write", content=raw)],
                issues=["Response was not valid JSON; applied as raw file content"],
            )

        return WorkerResult(
            summary=raw[:500],
            operations=[],
            issues=["Response was not valid JSON and could not be interpreted as file content"],
        )

    async def run(self) -> str:
        """Execute the task: read context, request operations, apply them.

        Returns a concise textual summary of what was changed, including
        counts of writes/edits and any issues encountered.
        """
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        context = await self._read_context()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n\n"
            f"--- File Context ---\n{context}\n--- End File Context ---\n"
        )
        await log.ainfo("worker.run", task_id=self.task.id, files=self.task.files)

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )
        result = self._parse_response(response)

        writes = 0
        edits = 0
        for op in result.operations:
            error = await self._apply_operation(op)
            if error is None:
                if op.operation == "write":
                    writes += 1
                else:
                    edits += 1
            else:
                result.issues.append(f"Failed to {op.operation} {op.path}: {error}")

        summary = f"Wrote {writes} files, edited {edits} files"
        if result.issues:
            summary += f", issues: [{'; '.join(result.issues)}]"
        return summary
