from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

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
        self.client = client or LLMClient(settings=settings)

    def _resolve_path(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str((self.client.settings.workspace / path).resolve())

    def _workspace_files(self) -> list[str]:
        try:
            return self.client.list_files(self.client.settings.workspace)
        except Exception as exc:
            return [f"[Could not list workspace: {exc}]"]

    def _files_to_read(self) -> list[str]:
        if self.task.files:
            return list(self.task.files)
        # No explicit files; surface up to 3 from the workspace root.
        all_files = self._workspace_files()
        relevant = [f for f in all_files if not f.startswith("[") and not f.startswith(".")]
        return relevant[:3]

    async def _build_file_context(self) -> str:
        lines: list[str] = []

        workspace_files = self._workspace_files()
        lines.append("--- Workspace file structure ---")
        lines.extend(workspace_files)
        lines.append("")

        for file_path in self._files_to_read():
            resolved = self._resolve_path(file_path)
            lines.append(f"--- File: {file_path} ---")
            try:
                content = await self.client.read_file(resolved)
                lines.append(content)
            except FileNotFoundError:
                lines.append(f"[File not found: {resolved}]")
            except Exception as exc:
                lines.append(f"[Could not read file: {exc}]")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_structured(data: dict) -> str:
        parts: list[str] = []
        changed = data.get("changed_files")
        if isinstance(changed, list) and changed:
            parts.append("Changed files: " + ", ".join(str(f) for f in changed))
        summary = data.get("summary")
        if isinstance(summary, str) and summary:
            parts.append(f"Summary: {summary}")
        issues = data.get("issues")
        if isinstance(issues, list) and issues:
            parts.append("Issues: " + ", ".join(str(i) for i in issues))
        return " | ".join(parts) if parts else ""

    @staticmethod
    def _parse_response(response: str) -> str:
        stripped = response.strip()
        candidates: list[str] = [stripped]
        # Allow responses wrapped in markdown code fences.
        fence_match = re.search(r"```(?:json)?\n(.*)\n```", stripped, re.DOTALL)
        if fence_match:
            candidates.append(fence_match.group(1).strip())
        # Allow a JSON object embedded somewhere in the response.
        json_match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if json_match:
            candidates.append(json_match.group(0).strip())
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(data, dict):
                formatted = WorkerAgent._format_structured(data)
                if formatted:
                    return formatted
        # Fallback: return the raw text response.
        return stripped

    async def run(self) -> str:
        context = await self._build_file_context()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"{context}"
            f"Task: {self.task.description}\n"
            f"Files to touch: "
            f"{', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )
        return self._parse_response(response)
