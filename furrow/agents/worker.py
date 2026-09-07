from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


@dataclass
class FileEdit:
    kind: str
    path: str
    content: str | None = None
    old_str: str | None = None
    new_str: str | None = None


_WRITE_RE = re.compile(
    r"<FURROW_WRITE><PATH>(.*?)</PATH><CONTENT>(.*?)</CONTENT></FURROW_WRITE>",
    re.DOTALL,
)

_STRREPLACE_RE = re.compile(
    r"<FURROW_STRREPLACE><PATH>(.*?)</PATH><OLD>(.*?)</OLD><NEW>(.*?)</NEW></FURROW_STRREPLACE>",
    re.DOTALL,
)

_SUMMARY_MARKER = "--- SUMMARY ---"


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.client.settings.workspace / p

    def _parse_edits(self, response: str) -> list[FileEdit]:
        edits: list[FileEdit] = []
        for match in _WRITE_RE.finditer(response):
            edits.append(FileEdit(kind="write", path=match.group(1).strip(), content=match.group(2)))
        for match in _STRREPLACE_RE.finditer(response):
            edits.append(FileEdit(kind="strreplace", path=match.group(1).strip(), old_str=match.group(2), new_str=match.group(3)))
        return edits

    def _extract_summary(self, response: str) -> str:
        idx = response.find(_SUMMARY_MARKER)
        if idx == -1:
            return ""
        return response[idx + len(_SUMMARY_MARKER):].strip()

    async def _apply_edits(self, edits: list[FileEdit]) -> list[str]:
        allowed = {Path(f).as_posix() for f in self.task.files} if self.task.files else set()
        results: list[str] = []
        for edit in edits:
            posix = Path(edit.path).as_posix()
            if allowed and posix not in allowed:
                results.append(f"FAILED {edit.kind} {posix}: not in allowed files")
                continue
            resolved = self._resolve(edit.path)
            try:
                if edit.kind == "write":
                    await self.client.write_file(resolved, edit.content or "")
                    results.append(f"APPLIED write {posix}")
                elif edit.kind == "strreplace":
                    await self.client.apply_str_replace(resolved, edit.old_str or "", edit.new_str or "")
                    results.append(f"APPLIED strreplace {posix}")
            except Exception as exc:
                results.append(f"FAILED {edit.kind} {posix}: {exc}")
        return results

    async def run(self) -> str:
        context_parts = []
        for f in self.task.files:
            content = await self.client.read_file_if_exists(f)
            if content is not None:
                context_parts.append(f"--- {f} ---\n{content}")

        file_context = "\n\n".join(context_parts) if context_parts else ""

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        if file_context:
            prompt += f"\nExisting file contents:\n{file_context}\n"

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        edits = self._parse_edits(response)
        results = await self._apply_edits(edits)
        summary = self._extract_summary(response)

        count = sum(1 for r in results if r.startswith("APPLIED"))
        lines = [f"Applied {count} edit(s)"]
        for r in results:
            lines.append(f"- {r}")
        if summary:
            lines.append("")
            lines.append(f"--- SUMMARY ---")
            lines.append(summary)

        return "\n".join(lines)
