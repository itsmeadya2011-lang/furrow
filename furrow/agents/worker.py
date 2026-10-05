"""Worker agent: implements one task and writes the resulting files to disk.

The worker is the only agent that mutates the repository. It hands the model the
current contents of the task's files, then applies the ``path:`` fenced blocks
that come back, refusing anything that resolves outside the workspace.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

__all__ = ["WorkerAgent", "parse_file_blocks", "resolve_in_workspace"]

log = structlog.get_logger(__name__)

_FENCE = "```"
_PATH_PREFIX = "path:"
#: Characters kept from the model's own prose when building the summary.
_PROSE_LIMIT = 400
#: Hard ceiling on the string returned by :meth:`WorkerAgent.run`.
_MAX_SUMMARY = 1200
#: Defaults matching the documented :class:`~furrow.config.Settings` fields.
_DEFAULT_FILE_CONTEXT_CHARS = 4000
_DEFAULT_MAX_PROMPT_FILES = 200

_UNSAFE_CHARS = ('"', "'", "`", "<", ">", "|", "*", "?", ":")
_WHITESPACE = re.compile(r"\s+")


class WorkerAgent:
    """Implement a single :class:`~furrow.config.TaskModel` and persist changes."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Bind the task to a client.

        Args:
            task: The task to implement.
            client: Shared :class:`~furrow.llm.LLMClient`; created if omitted.
            settings: Explicit settings; defaults to the client's own settings.
        """
        self.task = task
        if client is None:
            self.client = LLMClient(settings=settings) if settings is not None else LLMClient()
        else:
            self.client = client
        self.settings = settings if settings is not None else getattr(self.client, "settings", None)

    async def run(self) -> str:
        """Implement the task, writing any returned file blocks.

        Returns:
            A short plain-text summary (under ~1200 characters) listing what
            changed, which files were written, which paths were rejected and
            why, and any unresolved issues. Never raises for a rejected path or
            a failed individual write.
        """
        workspace = self._workspace()
        listing = self._listing() if not self.task.files else []
        contents = await self._read_context()
        prompt = self._build_prompt(listing, contents)

        response = await self.client.complete(
            prompt,
            system=WORKER_SYSTEM,
            model=self._setting("worker_model", None),
        )

        written, rejected, issues = await self._apply_blocks(parse_file_blocks(response), workspace)
        return self._summarise(response, written, rejected, issues)

    async def _apply_blocks(
        self,
        blocks: list[tuple[str, str]],
        workspace: Path,
    ) -> tuple[list[str], list[str], list[str]]:
        """Write every safe block; collect written paths, rejects and issues."""
        written: list[str] = []
        rejected: list[str] = []
        issues: list[str] = []

        for path, content in blocks:
            target = resolve_in_workspace(workspace, path)
            if target is None:
                rejected.append(f"{path}: unsafe path (must stay inside the workspace)")
                log.warning("worker_path_rejected", path=path)
                continue
            relative = _relative(target, workspace)
            try:
                await self.client.write_file(target, content)
            except (OSError, ValueError) as exc:
                message = f"{relative}: write failed ({type(exc).__name__}: {exc})"
                issues.append(message)
                log.warning("worker_write_failed", path=relative, error=str(exc)[:300])
                continue
            written.append(relative)
            log.info("worker_file_written", path=relative, chars=len(content))

        return written, rejected, issues

    async def _read_context(self) -> dict[str, str]:
        """Read the declared files that exist; unreadable paths are simply absent."""
        declared = [str(path) for path in getattr(self.task, "files", None) or []]
        if not declared:
            return {}
        limit = int(self._setting("file_context_chars", _DEFAULT_FILE_CONTEXT_CHARS))
        try:
            mapping = await self.client.read_files(declared, max_chars=limit)
        except OSError as exc:
            log.warning("worker_context_unreadable", error=str(exc)[:300])
            return {}
        contents: dict[str, str] = {}
        for path in declared:
            key = _key(mapping, path)
            if key is not None:
                contents[path] = str(mapping[key])
        return contents

    def _build_prompt(self, listing: list[str], contents: dict[str, str]) -> str:
        """Assemble instructions, task, listing and file contents into one prompt."""
        task = self.task
        dependencies = [str(item) for item in getattr(task, "dependencies", None) or []]
        sections = [
            WORKER_PROMPT,
            f"\n\nTask id: {task.id}",
            f"\nTask: {task.description}",
            "\nDepends on: " + (", ".join(dependencies) if dependencies else "nothing"),
        ]

        if listing:
            sections.append(
                "\n\nRepository files (relative paths, pick your targets from these):\n"
                + "\n".join(listing)
            )
        elif task.files:
            sections.append(
                "\n\nDeclared files for this task:\n" + "\n".join(str(f) for f in task.files)
            )

        if contents:
            blocks = []
            for path, text in contents.items():
                blocks.append(f"--- BEGIN FILE: {path} ---\n{text}\n--- END FILE: {path} ---")
            sections.append("\n\nExisting file contents:\n" + "\n\n".join(blocks))
        else:
            sections.append(
                "\n\nExisting file contents: none of the declared files exist yet; "
                "create them."
            )

        return "".join(sections)

    def _listing(self) -> list[str]:
        """List workspace files so a task without declared files can pick targets."""
        try:
            files = list(self.client.list_files(self._workspace()))
        except OSError:
            return []
        max_files = int(self._setting("max_prompt_files", _DEFAULT_MAX_PROMPT_FILES))
        return files[:max_files] if max_files > 0 else files

    def _summarise(
        self,
        response: str,
        written: list[str],
        rejected: list[str],
        issues: list[str],
    ) -> str:
        """Build the compact task-result string returned by :meth:`run`."""
        prose = _strip_fences(response)
        headline = _WHITESPACE.sub(" ", prose).strip()
        if len(headline) > _PROSE_LIMIT:
            headline = headline[:_PROSE_LIMIT] + "..."
        lines = [headline or "Worker returned no summary text.", f"Files written: {_join(written)}"]
        if rejected:
            lines.append("Rejected: " + _join(rejected))
        if issues:
            lines.append("Unresolved: " + _join(issues))
        text = "\n".join(lines)
        if len(text) > _MAX_SUMMARY:
            text = text[:_MAX_SUMMARY].rstrip() + "\n... [truncated]"
        return text

    def _workspace(self) -> Path:
        """Return the resolved workspace root as a :class:`~pathlib.Path`."""
        return Path(self._setting("workspace", Path.cwd())).resolve()

    def _setting(self, name: str, default: Any) -> Any:
        """Read a setting from the explicit settings or the client's settings."""
        holder = self.settings if self.settings is not None else getattr(
            self.client, "settings", None
        )
        value = getattr(holder, name, default)
        return default if value is None else value


def parse_file_blocks(response: str) -> list[tuple[str, str]]:
    """Extract ``(relative_path, content)`` pairs from a worker response.

    Only two fence forms are accepted:

    * an info string starting with ``path:`` (case-insensitive), optionally
      preceded by a language tag, e.g. ```` ```path: src/a.py ```` or
      ```` ```python path: src/a.py ````;
    * an info string that is itself a plausible relative path (no spaces, and
      containing a ``/`` or a ``.``), e.g. ```` ```src/a.py ````.

    Fences with any other info string (``bash``, ``json``, ``diff``, ...) are
    ignored, as are unterminated path fences.

    Args:
        response: The raw model output.

    Returns:
        One tuple per accepted block, in document order. The content is the
        full new file body, newline-terminated.
    """
    if not isinstance(response, str):
        return []

    lines = response.splitlines()
    blocks: list[tuple[str, str]] = []
    index = 0
    total = len(lines)

    while index < total:
        stripped = lines[index].strip()
        if not stripped.startswith(_FENCE):
            index += 1
            continue
        info = stripped[len(_FENCE) :].strip()
        end = index + 1
        while end < total and not lines[end].strip().startswith(_FENCE):
            end += 1
        path = _path_from_info(info)
        if path:
            body = "".join(f"{line}\n" for line in lines[index + 1 : end])
            blocks.append((path, body))
        index = end + 1 if end < total else total

    return blocks


def resolve_in_workspace(root: Path, candidate: str) -> Path | None:
    """Resolve ``candidate`` under ``root``, returning ``None`` if it escapes.

    Args:
        root: The workspace root the path must stay inside.
        candidate: A repository-relative path emitted by the model.

    Returns:
        The absolute, normalised path, or ``None`` when the candidate is empty,
        absolute, home-relative, contains ``..``, or resolves outside ``root``.
    """
    raw = (candidate or "").strip()
    if not raw or raw.startswith("~"):
        return None
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    try:
        base = Path(root).resolve()
        target = (base / relative).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    return target if target.is_relative_to(base) else None


def _path_from_info(info: str) -> str | None:
    """Return the path encoded in a fence info string, or ``None``."""
    tokens = info.split()
    if not tokens:
        return None

    for position, token in enumerate(tokens):
        lowered = token.lower()
        if not lowered.startswith(_PATH_PREFIX):
            continue
        remainder = token[len(_PATH_PREFIX) :].strip()
        if remainder:
            return remainder
        return " ".join(tokens[position + 1 :]).strip() or None

    if len(tokens) == 1 and _looks_like_path(tokens[0]):
        return tokens[0]
    return None


def _looks_like_path(token: str) -> bool:
    """Report whether ``token`` plausibly is a relative path rather than a language tag."""
    if not token or any(char.isspace() for char in token):
        return False
    if any(char in token for char in _UNSAFE_CHARS):
        return False
    return "/" in token or "." in token


def _strip_fences(response: str) -> str:
    """Return ``response`` with every fenced block removed, for prose reuse."""
    lines = response.splitlines()
    kept: list[str] = []
    inside = False
    for line in lines:
        if line.strip().startswith(_FENCE):
            inside = not inside
            continue
        if not inside:
            kept.append(line)
    return "\n".join(kept)


def _relative(target: Path, root: Path) -> str:
    """Express ``target`` relative to ``root``, falling back to the full path."""
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return str(target)


def _key(mapping: dict[str, str], path: str) -> str | None:
    """Find the mapping key that corresponds to ``path``, if any."""
    for candidate in (path, Path(path).as_posix()):
        if candidate in mapping:
            return candidate
    return None


def _join(items: list[str]) -> str:
    """Render a list compactly, defaulting to ``none`` when empty."""
    return ", ".join(items) if items else "none"