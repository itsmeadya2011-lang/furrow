from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Read existing file contents so the LLM can edit them in place.
        file_contents: dict[str, str] = {}
        for file_path in self.task.files:
            p = Path(file_path)
            if p.exists():
                file_contents[file_path] = await self.client.read_file(p)

        prompt = self._build_prompt(file_contents)
        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: treat the raw response as a summary, no file changes.
            return response.strip()

        files = data.get("files", [])
        summary = data.get("summary", "No summary provided.")

        written: list[str] = []
        for op in files:
            path = op.get("path") or op.get("file")
            content = op.get("content", "")
            if not path:
                continue
            await self.client.write_file(path, content)
            written.append(path)

        if written:
            return f"{summary}\nFiles written: {', '.join(written)}"
        return summary

    def _build_prompt(self, file_contents: dict[str, str]) -> str:
        files_section = ""
        if file_contents:
            parts = []
            for path, content in file_contents.items():
                parts.append(f"--- {path} ---\n{content}\n--- end {path} ---")
            files_section = "\nExisting file contents:\n" + "\n".join(parts)

        files_list = ", ".join(self.task.files) if self.task.files else "none specified"

        return (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_list}"
            f"{files_section}\n\n"
            "Return JSON only (no markdown) with this shape:\n"
            "{\n"
            '  "files": [\n'
            '    {"path": "src/foo.py", "content": "full new file content"}\n'
            "  ],\n"
            '  "summary": "Brief summary of changes"\n'
            "}\n"
            "If no file changes are needed, return an empty files array and a summary explaining why."
        )
