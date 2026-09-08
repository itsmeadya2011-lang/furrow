from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)

FILE_BLOCK_RE = re.compile(r'---FILE:\s*(.+?)---\s*(.*?)---END FILE---', re.DOTALL)
SUMMARY_RE = re.compile(r'SUMMARY:\s*(.+)', re.MULTILINE)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def _read_task_files(self) -> dict[str, str]:
        contents: dict[str, str] = {}
        for file_path in self.task.files:
            try:
                contents[file_path] = await self.client.read_file(file_path)
            except FileNotFoundError:
                logger.warning("File %s does not exist; skipping read", file_path)
            except OSError as exc:
                logger.warning("Could not read %s: %s; skipping", file_path, exc)
        return contents

    async def run(self) -> str:
        file_contents = await self._read_task_files()

        lines = [f"{WORKER_PROMPT}\n"]
        lines.append(f"Task description: {self.task.description}")
        lines.append("Files to touch: " + (", ".join(self.task.files) if self.task.files else "any"))
        if file_contents:
            lines.append("\nExisting file contents:")
            for path, content in file_contents.items():
                lines.append(f"---FILE: {path}---")
                lines.append(content)
                lines.append("---END FILE---")
        prompt = "\n".join(lines)

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        file_blocks = FILE_BLOCK_RE.findall(response)
        written: list[str] = []
        for path_str, content in file_blocks:
            file_path = Path(path_str.strip())
            await self.client.write_file(file_path, content)
            written.append(str(file_path))

        summary_match = SUMMARY_RE.search(response)
        summary_line = summary_match.group(1).strip() if summary_match else ""

        written_str = ", ".join(written) if written else "none"
        summary_str = f" Summary: {summary_line}." if summary_line else ""
        return f"Updated {len(written)} files: {written_str}.{summary_str}"
