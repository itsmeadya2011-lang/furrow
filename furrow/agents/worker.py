from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_FILE_CHARS = 8000
MAX_CONTEXT_FILES = 10
MAX_REPO_FILES = 200


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding ```json / ``` code fence if present."""
    text = text.strip()
    if text.startswith("```"):
        _, _, rest = text.partition("\n")
        rest = rest.removeprefix("```")
        if "```" in rest:
            rest = rest.rsplit("```", 1)[0]
        return rest.strip()
    return text


def _first_balanced_json_object(text: str) -> str | None:
    """Return the first balanced {...} JSON object substring in text, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\" and in_string:
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def _parse_worker_response(text: str) -> dict[str, Any]:
    """Extract the worker JSON patch from raw LLM output.

    Strips markdown code fences, locates the first balanced JSON object, and
    parses it. Raises ValueError if no machine-readable JSON is found.
    """
    for candidate in (_strip_code_fences(text), text):
        obj = _first_balanced_json_object(candidate)
        if obj is None:
            continue
        return json.loads(obj)
    raise ValueError("no machine-readable JSON object found in response")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    def _workspace(self) -> Path:
        workspace = getattr(self.client.settings, "workspace", None) or Path.cwd()
        return Path(workspace)

    def _resolve_target(self, path: str | Path, workspace: Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else workspace / p

    def _is_within_workspace(self, path: str | Path, workspace: Path) -> bool:
        p = Path(path)
        if ".." in p.parts:
            return False
        target = self._resolve_target(p, workspace)
        try:
            target.resolve().relative_to(workspace.resolve())
        except ValueError:
            return False
        return True

    async def run(self) -> str:
        workspace = self._workspace()

        # Gather current file contents; read errors are non-fatal (skip the file).
        current_contents: dict[str, str] = {}
        for path in self.task.files:
            try:
                current_contents[path] = await self.client.read_file(path)
            except (OSError, ValueError, KeyError, UnicodeDecodeError):
                continue

        # Repository map, bounded so the prompt stays small.
        try:
            repo_files = list(self.client.list_files(workspace))[:MAX_REPO_FILES]
        except (OSError, ValueError, KeyError, UnicodeDecodeError):
            repo_files = []
        repo_map = "\n".join(repo_files) if repo_files else "(empty)"

        # Bounded file context: truncate each file, skip extra files.
        blocks: list[str] = []
        for path in list(current_contents)[:MAX_CONTEXT_FILES]:
            content = current_contents[path]
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + "\n... (truncated) ..."
            blocks.append(f"### {path}\n```\n{content}\n```")
        file_context = "\n\n".join(blocks) if blocks else "(none)"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task description:\n{self.task.description}\n\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"Current file contents:\n{file_context}\n\n"
            f"Repository map (first {MAX_REPO_FILES} files):\n{repo_map}\n"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse the JSON patch; never crash on malformed LLM output.
        try:
            data = _parse_worker_response(response)
        except ValueError:
            snippet = response[:500]
            return (
                "No machine-readable patch was returned by the worker. "
                f"Raw response (truncated): {snippet!r}"
            )

        summary = str(data.get("summary", ""))
        files = data.get("files", {})
        if not isinstance(files, dict):
            files = {}

        # Apply the patch: reject escaping paths, write everything else.
        written: list[str] = []
        rejected: list[str] = []
        write_error: OSError | None = None
        for path, content in files.items():
            if not isinstance(path, str) or not isinstance(content, str):
                continue
            if not self._is_within_workspace(path, workspace):
                rejected.append(path)
                continue
            target = self._resolve_target(path, workspace)
            try:
                await self.client.write_file(target, content)
                written.append(str(target))
            except OSError as exc:
                # Keep writing remaining files so a partial patch is applied.
                if write_error is None:
                    write_error = exc

        # A write error must raise so the orchestrator can retry the task.
        if write_error is not None:
            raise write_error

        parts: list[str] = []
        if summary:
            parts.append(summary)
        if written:
            parts.append("wrote: " + ", ".join(written))
        if rejected:
            parts.append("rejected (outside workspace): " + ", ".join(rejected))
        return " | ".join(parts) if parts else "No changes made."
