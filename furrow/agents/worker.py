from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import render_worker_prompt
from furrow.config import TaskModel, get_logger
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger("furrow.worker")

# Pattern to match FILE: blocks in LLM responses.
# Matches: FILE: path/to/file\n```lang\n<content>\n```
FILE_BLOCK_PATTERN = re.compile(
    r"FILE:\s*(.+?)\n```(?:\w+)?\n(.*?)```",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings

    async def run(self) -> str:
        """Execute the assigned task.

        Reads existing files for context, sends the task to the LLM,
        and writes any file changes the LLM describes back to disk.
        """
        log.info("worker.run.start", task_id=self.task.id, description=self.task.description)

        # Gather context from existing files mentioned in the task
        context_parts: list[str] = []
        for file_path in self.task.files:
            full_path = self._resolve_path(file_path)
            try:
                content = await self.client.read_file(full_path)
                context_parts.append(
                    f"--- Existing file: {file_path} ---\n{content}\n--- End {file_path} ---"
                )
            except FileNotFoundError:
                context_parts.append(f"--- File: {file_path} (does not exist yet) ---")
            except OSError as e:
                context_parts.append(f"--- File: {file_path} (could not read: {e}) ---")

        files_context = (
            "\n".join(context_parts) if context_parts else "No existing files to inspect."
        )
        files_str = ", ".join(self.task.files) if self.task.files else "any"

        prompt = render_worker_prompt(
            workspace=str(self.settings.workspace),
            files_context=files_context,
            task_description=self.task.description,
            files=files_str,
        )

        try:
            response = await self.client.complete(prompt, model=self.settings.worker_model)
        except Exception as e:
            log.error("worker.run.error", task_id=self.task.id, error=str(e))
            raise

        # Parse and write files from the LLM response
        written_files = await self._write_files_from_response(response)

        if written_files:
            log.info("worker.run.wrote_files", task_id=self.task.id, files=written_files)
            summary = (
                f"Completed task {self.task.id}: {self.task.description}\n"
                f"Wrote {len(written_files)} file(s): {', '.join(written_files)}\n"
                f"LLM response excerpt: {response[:500]}..."
            )
        else:
            log.info("worker.run.no_files", task_id=self.task.id)
            summary = (
                f"Completed task {self.task.id}: {self.task.description}\n"
                f"Response: {response[:1000]}"
            )

        log.info("worker.run.done", task_id=self.task.id)
        return summary

    def _resolve_path(self, file_path: str) -> Path:
        """Resolve a file path relative to the workspace."""
        if file_path.startswith("/"):
            return Path(file_path)
        return self.settings.workspace / file_path

    async def _write_files_from_response(self, response: str) -> list[str]:
        """Parse FILE: blocks from LLM response and write them to disk."""
        written: list[str] = []

        for match in FILE_BLOCK_PATTERN.finditer(response):
            file_name = match.group(1).strip()
            content = match.group(2)
            full_path = self._resolve_path(file_name)
            try:
                await self.client.write_file(full_path, content)
                written.append(file_name)
            except OSError as e:
                log.error("worker.write_file.error", file=file_name, error=str(e))

        return written


__all__ = ["WorkerAgent"]
