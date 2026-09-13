from __future__ import annotations

import re
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
        workspace = Path(self.client.settings.workspace)

        # Read existing files for context
        existing_content = {}
        for file_path in self.task.files:
            full_path = workspace / file_path
            if full_path.exists():
                try:
                    content = await self.client.read_file(str(full_path))
                    existing_content[file_path] = content
                except Exception:
                    pass

        files_context = ""
        for path, content in existing_content.items():
            files_context += f"\n--- File: {path} ---\n{content}\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{files_context}"
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Try to extract and write files from the response
        written = self._extract_and_write_files(response, workspace)
        if written:
            return f"{response}\n[Wrote files: {', '.join(written)}]"
        return response

    def _extract_and_write_files(self, response: str, workspace: Path) -> list[str]:
        """Extract file writes from the LLM response using ---FILE: path--- markers."""
        pattern = r"---FILE:\s*(.+?)---\n(.*?)---END---"
        written = []
        for match in re.finditer(pattern, response, re.DOTALL):
            file_path = match.group(1).strip()
            content = match.group(2)
            try:
                full_path = workspace / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                written.append(file_path)
            except Exception as e:
                import structlog
                logger = structlog.get_logger()
                logger.error("file_write_failed", path=file_path, error=str(e))
        return written
