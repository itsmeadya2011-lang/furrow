from __future__ import annotations

import re
from pathlib import Path

from furrow.agents.base import BaseAgent
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

_FILE_HEADER = re.compile(r"^\s*FILE:\s*(?P<path>\S.*?)\s*$", re.IGNORECASE)


class WorkerAgent(BaseAgent):
    model_key = "worker_model"

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(client, settings)
        self.task = task

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(prompt, model=self.model)
        return await self._apply(parse_file_blocks(response), response)

    async def _apply(self, blocks: list[tuple[str, str]], response: str) -> str:
        """Write every block's content via the client. Writing files only, no execution."""
        root = self.workspace
        written: list[str] = []
        rejected: list[str] = []
        for raw_path, content in blocks:
            target = _safe_path(raw_path, root)
            if target is None:
                rejected.append(raw_path)
                continue
            await self.client.write_file(target, content)
            written.append(target.relative_to(root).as_posix())
        if written:
            summary = f"Wrote {len(written)} file(s): {', '.join(written)}."
        else:
            summary = "No file blocks in response, so nothing was written."
            summary += f" Model said: {_prose(response)}"
        if rejected:
            summary += f" Rejected {len(rejected)} unsafe path(s): {', '.join(rejected)}."
        return summary


def parse_file_blocks(text: str) -> list[tuple[str, str]]:
    """Parse `FILE: <path>` + fenced-block pairs: (path, full file content) in order."""
    blocks: list[tuple[str, str]] = []
    path: str | None = None
    fence = ""
    body: list[str] = []

    for line in text.splitlines():
        if fence:
            stripped = line.strip()
            if stripped and set(stripped) == {"`"} and len(stripped) >= len(fence):
                blocks.append((path or "", "\n".join(body) + "\n" if body else ""))
                path, fence, body = None, "", []
            else:
                body.append(line)
            continue
        header = _FILE_HEADER.match(line)
        if header:
            path = header.group("path")
        elif path and line.strip().startswith("```"):
            opened = line.strip()
            fence = opened[: len(opened) - len(opened.lstrip("`"))]
            body = []
    return blocks


def _safe_path(raw: str, root: Path) -> Path | None:
    """Resolve a model-authored path inside the workspace, or None if it escapes it."""
    rel = Path(raw.strip().strip("`").strip())
    if not rel.parts or rel.is_absolute() or ".." in rel.parts:
        return None
    target = (root / rel).resolve()
    return target if root in target.parents else None


def _prose(text: str, limit: int = 200) -> str:
    """First non-empty line of a response, for the no-blocks summary."""
    return next((line.strip() for line in text.splitlines() if line.strip()), "<no prose>")[:limit]
