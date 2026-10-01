from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

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

    async def run(self) -> str:
        files_context = await self._gather_file_context()
        prompt = self._build_prompt(files_context)
        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )
        parsed = self._parse_response(response)
        await self._write_files(parsed.get("files", []))
        summary = parsed.get("summary", "") or response
        return summary

    async def _gather_file_context(self) -> str:
        """Read existing files mentioned in the task so the LLM has context."""
        lines: list[str] = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                lines.append(f"--- {file_path} ---\n{content}\n")
            except (FileNotFoundError, OSError):
                pass
        return "\n".join(lines)

    def _build_prompt(self, files_context: str) -> str:
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        base = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {files_str}\n"
        if files_context.strip():
            base += f"\nExisting file contents for context:\n{files_context}\n"
        return base

    @staticmethod
    def _parse_response(response: str) -> dict[str, Any]:
        """Parse the LLM response into a structured dict.

        Handles both plain JSON and JSON wrapped in markdown code fences.
        Falls back to extracting file blocks from markdown if JSON parsing fails.
        """
        cleaned = response.strip()
        # Strip markdown code fences if present.
        if cleaned.startswith("```"):
            # Remove the opening fence line.
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1:]
            # Remove trailing fence.
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Fallback: parse markdown code blocks into file entries.
        files: list[dict[str, Any]] = []
        current_path: str | None = None
        current_lines: list[str] = []

        def _flush() -> None:
            if current_path is not None:
                files.append({"path": current_path, "content": "\n".join(current_lines)})

        for line in cleaned.splitlines():
            stripped = line.strip()
            if stripped.startswith("```") and "=" in stripped:
                # e.g. ```path=src/auth.py
                header = stripped.removeprefix("```").strip()
                eq_idx = header.find("=")
                if eq_idx != -1:
                    _flush()
                    current_path = header[eq_idx + 1:].strip()
                    current_lines = []
                    continue
            if stripped.startswith("```") and current_path is not None:
                _flush()
                current_path = None
                current_lines = []
                continue
            if current_path is not None:
                current_lines.append(line)

        _flush()
        return {"files": files, "summary": response}

    async def _write_files(self, files: list[dict[str, Any]]) -> None:
        """Write each file returned by the LLM to disk."""
        for entry in files:
            path = entry.get("path")
            content = entry.get("content", "")
            if not path:
                continue
            await self.client.write_file(path, content)
