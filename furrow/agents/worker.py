from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


def _extract_code_blocks(text: str) -> dict[str, str]:
    """Extract code blocks from LLM response.

    Matches ```filename.ext or ```lang patterns and extracts content.
    Falls back to treating the entire response as a single file if no blocks found.
    """
    blocks: dict[str, str] = {}
    pattern = re.compile(r"```(?:\w+\s+)?([^\n]+)?\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(text):
        filename = match.group(1) or "output.txt"
        content = match.group(2).strip()
        blocks[filename] = content
    if blocks:
        return blocks
    return {"output.txt": text.strip()}


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self._logger = logger.bind(component="worker", task_id=task.id)

    async def run(self) -> dict[str, Any]:
        """Execute the assigned task and return structured results.

        Returns:
            dict with keys: summary, files_written, issues
        """
        self._logger.info("task_start", description=self.task.description)
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"

        issues: list[str] = []
        files_written: list[str] = []

        try:
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        except Exception as e:
            self._logger.error("task_error", error=str(e))
            return {"summary": f"Task failed: {e}", "files_written": [], "issues": [str(e)]}

        self._logger.info("llm_response_received", response_length=len(response))

        # Read existing files if specified
        for filepath in self.task.files:
            try:
                content = await self.client.read_file(filepath)
                self._logger.info("file_read", path=filepath, size=len(content))
            except Exception as e:
                self._logger.warning("file_read_failed", path=filepath, error=str(e))
                issues.append(f"Could not read {filepath}: {e}")

        # Extract and write code blocks
        blocks = _extract_code_blocks(response)
        workspace = Path(self.client.settings.workspace).resolve()

        for filename, content in blocks.items():
            target = Path(filename)
            target_path = (workspace / target).resolve() if not target.is_absolute() else target
            if not str(target_path).startswith(str(workspace)):
                self._logger.warning("file_outside_workspace", path=str(target_path))
                issues.append(f"Refused to write outside workspace: {filename}")
                continue
            try:
                await self.client.write_file(target_path, content)
                files_written.append(str(target_path))
                self._logger.info("file_written", path=str(target_path), size=len(content))
            except Exception as e:
                self._logger.error("file_write_failed", path=str(target_path), error=str(e))
                issues.append(f"Could not write {filename}: {e}")

        summary = f"Completed task: {self.task.description}"
        if files_written:
            summary += f". Written to: {', '.join(files_written)}"
        if issues:
            summary += f". Issues: {'; '.join(issues)}"

        self._logger.info("task_complete", summary=summary)
        return {"summary": summary, "files_written": files_written, "issues": issues}
