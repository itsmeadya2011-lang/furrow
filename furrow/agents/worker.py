from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.planner import extract_json_object
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


def _resolve_workspace_path(raw_path: str, workspace: Path) -> Path | None:
    """Resolve *raw_path* inside *workspace*, or ``None`` if that escapes it.

    Absolute paths and paths that traverse outside the workspace are refused.
    """
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return None
    root = workspace.resolve()
    try:
        resolved = (root / candidate).resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(root):
        return None
    return resolved


def _safe_entries(raw_files: Any) -> list[dict]:
    """Return well-formed ``{"path": str, "content": str}`` file entries."""
    if not isinstance(raw_files, list):
        return []
    entries: list[dict] = []
    for raw in raw_files:
        if not isinstance(raw, dict):
            continue
        path = raw.get("path")
        content = raw.get("content")
        if not isinstance(path, str) or not path.strip():
            continue
        entries.append({"path": path, "content": content if isinstance(content, str) else ""})
    return entries


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    def _build_prompt(self) -> str:
        targets = ", ".join(self.task.files) if self.task.files else "any"
        return f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {targets}\n"

    async def run(self) -> str:
        prompt = self._build_prompt()
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = extract_json_object(response)
        except ValueError:
            # Unparseable response: return it untouched rather than raising.
            return response

        summary = data.get("summary")
        summary = summary.strip() if isinstance(summary, str) else ""
        entries = _safe_entries(data.get("files"))
        if not entries:
            return summary or response

        workspace = Path(self.client.settings.workspace)
        written: list[str] = []
        rejected: list[str] = []
        for entry in entries:
            resolved = _resolve_workspace_path(entry["path"], workspace)
            if resolved is None:
                rejected.append(entry["path"])
                continue
            try:
                await self.client.write_file(resolved, entry["content"])
            except OSError as e:
                rejected.append(f"{entry['path']} ({type(e).__name__}: {e})")
                continue
            written.append(entry["path"])

        parts: list[str] = []
        if summary:
            parts.append(summary)
        if written:
            parts.append("Wrote: " + ", ".join(written))
        if rejected:
            parts.append("Rejected unsafe path: " + ", ".join(rejected))
        if not parts:
            return response
        return "\n".join(parts)