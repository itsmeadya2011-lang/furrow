from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


# Matches fenced code blocks with an optional filename annotation:
#   ```python path/to/file.py
#   ...content...
#   ```
# or
#   ```path/to/file.py
#   ...content...
#   ```
_FILE_BLOCK_RE = re.compile(
    r"```(?:\w+\s+)?(?P<path>[^\s`]+\.[A-Za-z0-9]+)\n(?P<body>.*?)```",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        target_files = (
            ", ".join(self.task.files) if self.task.files else "auto-detect from output"
        )
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files you may touch: {target_files}\n\n"
            "Return your implementation as fenced code blocks of the form:\n"
            "```<lang> path/to/file.py\n<file contents>\n```\n"
            "One block per file. After the code blocks, give a one-paragraph summary."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        files_written = _write_files_from_response(response, self.task.files)
        summary = _strip_code_blocks(response).strip()
        if files_written:
            written = ", ".join(files_written)
            return f"{summary}\n\nFiles written: {written}"
        return summary


def _write_files_from_response(response: str, allowed: list[str]) -> list[str]:
    """Parse fenced code blocks from the LLM response and write them to disk.

    Only writes to paths that are listed in `allowed` (if `allowed` is non-empty).
    Paths are resolved relative to the configured workspace.
    """
    from furrow.config import settings as global_settings

    written: list[str] = []
    allowed_set = {str(Path(p)) for p in (allowed or [])}
    for match in _FILE_BLOCK_RE.finditer(response):
        rel = match.group("path").strip().strip("/")
        body = match.group("body")
        # Only write files that are in the allowed list (when one was provided).
        if allowed_set and rel not in allowed_set:
            continue
        target = (Path(global_settings.workspace) / rel).resolve()
        # Refuse to write outside the workspace.
        workspace_resolved = Path(global_settings.workspace).resolve()
        if workspace_resolved not in target.parents and target != workspace_resolved:
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
            written.append(rel)
        except OSError:
            continue
    return written


def _strip_code_blocks(response: str) -> str:
    return re.sub(r"```.*?```", "", response, flags=re.DOTALL)