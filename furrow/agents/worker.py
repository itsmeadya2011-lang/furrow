from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Marker the LLM uses to delimit file contents in its response.
FILE_MARKER = re.compile(r"^### FILE:\s*(.+?)\s*$", re.MULTILINE)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.written_files: list[str] = []

    async def run(self) -> str:
        # Gather existing file contents so the LLM can edit in context.
        file_contents: dict[str, str] = {}
        for file_path in self.task.files:
            try:
                file_contents[file_path] = await self.client.read_file(file_path)
            except FileNotFoundError:
                file_contents[file_path] = ""  # new file
            except OSError:
                file_contents[file_path] = ""

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
        if self.task.dependencies:
            prompt += f"\nDependencies: {', '.join(self.task.dependencies)}\n"
        if file_contents:
            prompt += "\nExisting files:\n"
            for path, content in file_contents.items():
                prompt += f"\n--- {path} ---\n{content}\n"

        prompt += (
            "\nWrite the full desired contents of each affected file using this format:\n"
            "### FILE: path/to/file\n<contents>\n### FILE: path/to/other\n<contents>\n"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Apply the file changes the LLM specified.
        written = self._apply_file_changes(response)
        self.written_files = written

        summary_lines = [f"Task {self.task.id}: {self.task.description}"]
        if written:
            summary_lines.append("Files written: " + ", ".join(written))
        else:
            summary_lines.append("No file changes applied.")
        return "\n".join(summary_lines)

    def _apply_file_changes(self, response: str) -> list[str]:
        """Parse ### FILE: markers from the LLM response and write each file."""
        written: list[str] = []
        matches = list(FILE_MARKER.finditer(response))
        for idx, match in enumerate(matches):
            path = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(response)
            content = response[start:end].strip("\n")
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                # Use the sync write for simplicity; files are small.
                Path(path).write_text(content)
                written.append(path)
            except OSError:
                # Best-effort; don't crash the worker over a single file.
                pass
        return written
