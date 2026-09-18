from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import FileOperation, TaskModel, WorkerResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_OPERATION_ERRORS = (
    AttributeError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
    Exception,
)


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    in_string = False
    quote = ""
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                in_string = False
                quote = ""
            continue
        if character == '"':
            in_string = True
            quote = character
            continue
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    if not candidates:
        raise ValueError("response does not contain a JSON object")

    def is_structured(candidate: dict[str, Any]) -> bool:
        return "task_id" in candidate and any(
            key in candidate for key in ("operations", "files", "changes")
        )

    def is_operational(candidate: dict[str, Any]) -> bool:
        return any(key in candidate for key in ("operations", "files", "changes"))

    structured = [candidate for candidate in candidates if is_structured(candidate)]
    if structured:
        return structured[-1]
    operational = [candidate for candidate in candidates if is_operational(candidate)]
    if operational:
        return operational[-1]
    return candidates[-1]


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        if settings is None and client is not None:
            settings = getattr(client, "settings", None)
        if settings is None:
            settings = default_settings
        self.settings = settings
        self.workspace = Path(settings.workspace).expanduser().resolve()
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> WorkerResult:
        files_text = ", ".join(self.task.files) if self.task.files else "any"
        context = await self._read_context()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task ID: {self.task.id}\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_text}\n"
            f"Workspace: {self.workspace}\n"
        )
        if context:
            prompt += f"\nRelevant file context:\n{context}\n"

        try:
            response = await self.client.complete(
                prompt, model=self._worker_model()
            )
        except _OPERATION_ERRORS as exc:
            return self._failure_result(
                error="llm_error",
                summary="Worker execution could not contact the model.",
                issues=[self._short_error(exc)],
            )

        try:
            data = _extract_json_object(response if isinstance(response, str) else "")
        except (TypeError, ValueError) as exc:
            return self._failure_result(
                error="invalid_json",
                summary="Worker response could not be parsed.",
                issues=[self._short_error(exc)],
            )

        if data.get("task_id") != self.task.id:
            received = repr(data.get("task_id")) if "task_id" in data else "missing"
            return self._failure_result(
                error="task_id_mismatch",
                summary="Worker response was rejected.",
                issues=[f"task_id must be {self.task.id!r}; received {received}"],
            )

        if not any(key in data for key in ("operations", "files", "changes")):
            return self._failure_result(
                error="invalid_result",
                summary="Worker response did not describe file operations.",
                issues=["worker response is missing an operations list"],
            )

        try:
            parsed = WorkerResult.model_validate(data)
        except ValidationError as exc:
            return self._failure_result(
                error="invalid_result",
                summary="Worker response did not match the result schema.",
                issues=[self._short_error(exc)],
            )

        if parsed.success is False or parsed.error:
            issues = list(parsed.issues)
            if parsed.error:
                issues.append(parsed.error)
            return self._failure_result(
                error=parsed.error or "worker_declared_failure",
                summary=parsed.summary or "Worker reported failure.",
                issues=issues,
            )

        prepared, issues = await self._prepare_operations(parsed.operations)
        if issues:
            return self._failure_result(
                error="invalid_operations",
                summary=parsed.summary or "Worker operations were rejected.",
                issues=[*parsed.issues, *issues],
            )

        changed, issues = await self._apply_operations(prepared)
        success = not issues
        return WorkerResult(
            task_id=self.task.id,
            summary=parsed.summary or self._default_summary(changed),
            operations=changed,
            success=success,
            error=None if success else "operation_failed",
            issues=self._dedupe([*parsed.issues, *issues]),
            files_changed=self._changed_paths(changed),
        )

    async def _read_context(self) -> str:
        if not self.task.files:
            return ""

        blocks: list[str] = []
        for relative_path in self.task.files:
            try:
                target, _ = self._workspace_target(relative_path)
            except (OSError, RuntimeError, ValueError) as exc:
                blocks.append(f"--- {relative_path} ---\n[invalid path: {self._short_error(exc)}]")
                continue
            try:
                content = await self.client.read_file(target)
            except _OPERATION_ERRORS as exc:
                blocks.append(f"--- {relative_path} ---\n[unreadable: {self._short_error(exc)}]")
                continue
            blocks.append(f"--- {relative_path} ---\n{content}")
        return "\n\n".join(blocks)

    def _worker_model(self) -> str:
        client_settings = getattr(self.client, "settings", None)
        model = getattr(client_settings, "worker_model", None)
        if model is None:
            model = getattr(self.settings, "worker_model", None)
        return model or default_settings.worker_model

    async def _prepare_operations(
        self, operations: list[FileOperation]
    ) -> tuple[list[tuple[FileOperation, Path, str]], list[str]]:
        prepared: list[tuple[FileOperation, Path, str]] = []
        issues: list[str] = []
        seen_paths: set[Path] = set()

        for operation in operations:
            try:
                target, resolved = self._workspace_target(operation.path)
            except (OSError, RuntimeError, ValueError) as exc:
                issues.append(f"{operation.action} {operation.path}: {self._short_error(exc)}")
                continue

            if resolved in seen_paths:
                issues.append(f"{operation.action} {operation.path}: duplicate operation path")
                continue
            seen_paths.add(resolved)

            try:
                if operation.action == "write":
                    issue = await self._prepare_write(target, operation)
                else:
                    issue = self._prepare_delete(target)
            except _OPERATION_ERRORS as exc:
                issue = f"could not validate operation: {self._short_error(exc)}"
            if issue:
                issues.append(f"{operation.action} {operation.path}: {issue}")
                continue
            prepared.append((operation, target, operation.action))

        return prepared, self._dedupe(issues)

    async def _prepare_write(
        self, target: Path, operation: FileOperation
    ) -> str | None:
        if operation.content is None:
            return "write operation requires content"
        if target.is_symlink() or not os.path.lexists(str(target)):
            if target.is_symlink():
                return "symbolic links are not allowed"
            return None
        if not target.is_file():
            return "target is not a regular file"
        try:
            current = await self.client.read_file(target)
        except (FileNotFoundError, IsADirectoryError, OSError, PermissionError) as exc:
            return f"could not read existing file: {self._short_error(exc)}"
        if current == operation.content:
            return "no-op write operation would not change the file"
        return None

    def _prepare_delete(self, target: Path) -> str | None:
        if target.is_symlink():
            return "symbolic links are not allowed"
        if not os.path.lexists(str(target)):
            return "no-op delete operation has no target"
        if not target.is_file():
            return "delete target is not a regular file"
        return None

    async def _apply_operations(
        self, prepared: list[tuple[FileOperation, Path, str]]
    ) -> tuple[list[FileOperation], list[str]]:
        changed: list[FileOperation] = []
        issues: list[str] = []
        for operation, target, action in prepared:
            try:
                if action == "write":
                    await self._atomic_write(target, operation.content or "")
                else:
                    await self._safe_delete(target)
                changed.append(operation)
            except _OPERATION_ERRORS as exc:
                issues.append(
                    f"{action} {operation.path} failed: {self._short_error(exc)}"
                )
        return changed, self._dedupe(issues)

    async def _atomic_write(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved = target.resolve()
        self._ensure_within_workspace(resolved)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            await self.client.write_file(temporary, content)
            resolved = target.resolve()
            self._ensure_within_workspace(resolved)
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    async def _safe_delete(self, target: Path) -> None:
        resolved = target.resolve()
        self._ensure_within_workspace(resolved)
        if target.is_symlink() or not target.is_file():
            raise ValueError("delete target is not a regular file")
        target.unlink()

    def _workspace_target(self, relative_path: str) -> tuple[Path, Path]:
        normalized = relative_path.replace("\\", "/")
        if (
            "\x00" in normalized
            or any(ord(character) < 32 for character in normalized)
            or normalized.startswith(("/", "\\\\", "//"))
            or PureWindowsPath(relative_path).drive
        ):
            raise ValueError("path must be relative to the workspace")

        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or not path.parts
            or any(part == ".." for part in path.parts)
            or normalized.endswith("/")
            or normalized.endswith("/.")
        ):
            raise ValueError("path must identify a file within the workspace")

        target = self.workspace.joinpath(*path.parts)
        resolved = target.resolve()
        self._ensure_within_workspace(resolved)
        return target, resolved

    def _ensure_within_workspace(self, target: Path) -> None:
        try:
            target.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError("path escapes workspace") from exc

    def _failure_result(
        self,
        *,
        error: str,
        summary: str,
        issues: list[str],
    ) -> WorkerResult:
        return WorkerResult(
            task_id=self.task.id,
            summary=summary,
            operations=[],
            success=False,
            error=error,
            issues=self._dedupe(issues),
        )

    def _default_summary(self, changed: list[FileOperation]) -> str:
        if not changed:
            return "No file operations applied."
        writes = sum(operation.action == "write" for operation in changed)
        deletes = sum(operation.action == "delete" for operation in changed)
        if writes and deletes:
            return (
                f"Wrote {writes} file{'s' if writes != 1 else ''} and "
                f"deleted {deletes} file{'s' if deletes != 1 else ''}."
            )
        if writes:
            return f"Wrote {writes} file{'s' if writes != 1 else ''}."
        return f"Deleted {deletes} file{'s' if deletes != 1 else ''}."

    def _changed_paths(self, operations: list[FileOperation]) -> list[str]:
        return [operation.path.replace("\\", "/") for operation in operations]

    def _short_error(self, exc: BaseException) -> str:
        return " ".join(str(exc).split())[:300]

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value and value not in seen:
                seen.add(value)
                result.append(value)
        return result
