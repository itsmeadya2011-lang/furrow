from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


MAX_FILE_CONTENT_BYTES = 16 * 1024
MAX_WORKSPACE_FILES = 100
MAX_WORKSPACE_LISTING_CHARS = 12 * 1024


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings if settings is not None else getattr(self.client, "settings", None)

        workspace_value = getattr(self.settings, "workspace", None)
        try:
            self.workspace = Path(workspace_value).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            self.workspace = Path.cwd()

    def _resolve_assigned_file(self, workspace: Path, assigned_path: str) -> Path | None:
        try:
            path = Path(assigned_path)
            if not path.is_absolute():
                path = workspace / path
            resolved = path.resolve()
            resolved.relative_to(workspace)
        except (OSError, RuntimeError, ValueError):
            return None
        return resolved

    def _read_bounded_file(self, path: Path) -> str:
        try:
            with path.open("rb") as file:
                content = file.read(MAX_FILE_CONTENT_BYTES + 1)
        except OSError as exc:
            return f"[unable to read file: {exc}]"

        truncated = len(content) > MAX_FILE_CONTENT_BYTES
        text = content[:MAX_FILE_CONTENT_BYTES].decode("utf-8", errors="replace")
        if truncated:
            text += f"\n\n[content truncated after {MAX_FILE_CONTENT_BYTES} bytes]"
        return text

    def _list_workspace_files(self, workspace: Path) -> list[str]:
        files: list[str] = []
        listing_chars = 0
        try:
            for path in workspace.rglob("*"):
                if path.is_file():
                    relative = path.relative_to(workspace).as_posix()
                    if files and listing_chars + len(relative) + 1 > MAX_WORKSPACE_LISTING_CHARS:
                        break
                    files.append(relative)
                    listing_chars += len(relative) + 1
                    if len(files) >= MAX_WORKSPACE_FILES:
                        break
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return [f"[unable to list workspace: {exc}]"]
        return files

    def _workspace_context(self) -> str:
        workspace = self.workspace
        context = [f"Workspace: {workspace}"]
        assigned_files = self.task.files or []

        if assigned_files:
            context.append("Assigned file contents:")
            for assigned_path in assigned_files:
                resolved = self._resolve_assigned_file(workspace, assigned_path)
                if resolved is None:
                    context.append(f"- {assigned_path}: REJECTED (path is outside the workspace)")
                    continue
                if not resolved.exists():
                    context.append(f"- {assigned_path}: MISSING")
                    continue
                if not resolved.is_file():
                    context.append(f"- {assigned_path}: NOT A REGULAR FILE")
                    continue

                context.append(f"--- {assigned_path} ---")
                context.append(self._read_bounded_file(resolved))
        else:
            context.append(f"Workspace file listing (bounded to {MAX_WORKSPACE_FILES} files):")
            files = self._list_workspace_files(workspace)
            if files:
                context.extend(f"- {path}" for path in files)
            else:
                context.append("- (no files found)")

        return "\n".join(context)

    async def run(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n\n"
            f"Workspace context:\n{self._workspace_context()}\n"
        )
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
