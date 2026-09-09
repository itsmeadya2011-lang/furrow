from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    ) -> None:
        self.task = task
        self.client = client or (
            LLMClient(settings=settings) if settings is not None else LLMClient()
        )
        self.settings = settings or getattr(self.client, "settings", None)
        workspace = getattr(self.settings, "workspace", None) or Path.cwd()
        self.workspace = Path(workspace).expanduser()

    async def run(self) -> str:
        existing_files = await self._read_task_files()
        assigned_files = ", ".join(self.task.files) if self.task.files else "none"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n\n"
            f"Workspace: {self.workspace}\n\n"
            f"Assigned files: {assigned_files}\n\n"
            f"Existing file contents:\n{existing_files}\n"
        )
        client_settings = getattr(self.client, "settings", None) or self.settings
        worker_model = getattr(client_settings, "worker_model", None)
        if not isinstance(worker_model, str):
            response = await self.client.complete(prompt)
        else:
            response = await self.client.complete(prompt, model=worker_model)
        if not isinstance(response, str):
            response = str(response) if response is not None else ""
        operations = self._parse_operations(response)
        if operations is None:
            return response.strip() or "Worker returned no structured file operations."

        summary = await self._apply_operations(operations)
        return summary or response.strip() or "No file changes were requested."

    async def _read_task_files(self) -> str:
        if not self.task.files:
            return "No files were assigned. Create new files with write operations."

        sections: list[str] = []
        for path in self.task.files:
            try:
                content = await self.client.read_file(self._resolve_path(path))
                sections.append(f"--- {path} ---\n{content}")
            except Exception as exc:
                sections.append(
                    f"--- {path} ---\n"
                    f"<unable to read: {type(exc).__name__}: {exc}>"
                )
        return "\n\n".join(sections)

    @staticmethod
    def _parse_operations(response: str) -> list[dict[str, Any]] | None:
        text = response.strip()
        if not text:
            return None

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            start = text.find("[")
            end = text.rfind("]")
            if start < 0 or end <= start:
                return None
            try:
                data = json.loads(text[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                return None

        if not isinstance(data, list):
            return None

        operations: list[dict[str, Any]] = []
        for operation in data:
            if not isinstance(operation, dict):
                return None
            action = operation.get("action")
            path = operation.get("path")
            if (
                action not in {"read", "write"}
                or not isinstance(path, str)
                or not path.strip()
            ):
                return None
            if action == "write" and not isinstance(
                operation.get("content"), str
            ):
                return None
            operations.append(operation)
        return operations

    async def _apply_operations(self, operations: list[dict[str, Any]]) -> str:
        changed: list[str] = []
        read: list[str] = []
        errors: list[str] = []

        for operation in operations:
            action = str(operation["action"])
            path = str(operation["path"])
            try:
                resolved_path = self._resolve_path(path)
                if action == "write":
                    await self.client.write_file(
                        resolved_path,
                        str(operation["content"]),
                    )
                    changed.append(path)
                else:
                    await self.client.read_file(resolved_path)
                    read.append(path)
            except Exception as exc:
                errors.append(f"{action} {path}: {type(exc).__name__}: {exc}")

        parts: list[str] = []
        if changed:
            parts.append(f"Changed: {', '.join(changed)}")
        if read:
            parts.append(f"Read: {', '.join(read)}")
        if errors:
            parts.append(f"Errors: {'; '.join(errors)}")
        if not parts:
            return "No file changes were requested."
        return ". ".join(parts) + "."

    def _resolve_path(self, path: str) -> Path:
        workspace = self.workspace.resolve()
        requested = Path(path)
        if requested.is_absolute():
            raise ValueError(f"File path must be relative to the workspace: {path}")
        resolved = (workspace / requested).resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError:
            raise ValueError(f"File path escapes the workspace: {path}") from None
        return resolved
