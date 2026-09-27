from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_BLOCK_RE = re.compile(r"^\s*(FILE|SUMMARY):\s*(.*?)\s*$", re.MULTILINE)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    @property
    def _workspace(self):
        return self.client.settings.workspace

    async def _read_context(self) -> tuple[list[str], dict[str, str]]:
        workspace = self._workspace
        all_files = self.client.list_files(workspace)
        existing: dict[str, str] = {}
        for rel in self.task.files:
            try:
                existing[rel] = await self.client.read_file(workspace / rel)
            except FileNotFoundError:
                continue
        return all_files, existing

    async def _build_prompt(self, all_files: list[str], existing: dict[str, str]) -> str:
        files_hint = ", ".join(self.task.files) if self.task.files else "any"
        lines: list[str] = [WORKER_PROMPT.strip()]
        lines.append(f"Task: {self.task.description}")
        lines.append(f"Files to touch: {files_hint}")
        lines.append("")
        if all_files:
            lines.append("Workspace file structure:")
            lines.extend(f"  - {f}" for f in sorted(all_files))
        else:
            lines.append("Workspace file structure: (empty)")
        lines.append("")
        if existing:
            lines.append("Existing file contents to consider:")
            for rel, content in existing.items():
                lines.append(f"--- Existing: {rel} ---")
                lines.append(content)
                lines.append(f"--- End {rel} ---")
        else:
            lines.append("No existing task files found.")
        lines.append("")
        lines.append("Return your changes in this exact format. Each marker starts a new line:")
        lines.append("")
        lines.append("FILE: <relative_path>")
        lines.append("<file content>")
        lines.append("")
        lines.append("FILE: <relative_path>")
        lines.append("<file content>")
        lines.append("")
        lines.append("SUMMARY:")
        lines.append("<concise summary of what changed>")
        lines.append("")
        lines.append("Write file contents verbatim. Only include files you want to create or modify.")
        return "\n".join(lines)

    async def _parse_response(self, response: str) -> str:
        files: list[tuple[str, str]] = []
        summary_parts: list[str] = []

        markers = list(_BLOCK_RE.finditer(response))
        for i, match in enumerate(markers):
            marker = match.group(1).upper()
            header = match.group(2)
            start = match.end()
            end = markers[i + 1].start() if i + 1 < len(markers) else len(response)
            body = response[start:end].strip("\n")
            if marker == "FILE":
                path = header.strip()
                if path:
                    files.append((path, body))
            elif marker == "SUMMARY":
                if header:
                    summary_parts.append(header)
                if body:
                    summary_parts.append(body)

        if files:
            for rel_path, content in files:
                await self.client.write_file(self._workspace / rel_path, content)
            summary = "\n".join(summary_parts).strip()
            return summary or f"Applied {len(files)} file change(s)."

        return response.strip()

    async def run(self) -> str:
        all_files, existing = await self._read_context()
        prompt = await self._build_prompt(all_files, existing)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._parse_response(response)
