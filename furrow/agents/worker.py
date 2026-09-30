from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_MAX_SUMMARY_LEN = 2000


class WorkerAgent:
    """Executes a single planned task by editing files via the LLM.

    The worker reads existing files mentioned in the task for context,
    sends them to the LLM along with the task description, and parses the
    JSON response to write files.
    """

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Gather existing file contents for context
        file_context = await self._build_file_context()

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"\nCurrent file contents:\n{file_context}\n"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        return await self._apply_edits(response)

    async def _build_file_context(self) -> str:
        """Read existing files so the LLM has full context before editing."""
        if not self.task.files:
            return "(no specific files — create new files as needed)"
        parts: list[str] = []
        for rel_path in self.task.files:
            try:
                content = await self.client.read_file(rel_path)
            except (FileNotFoundError, OSError):
                content = "<file does not exist yet>"
            parts.append(f"--- {rel_path} ---\n{content}")
        return "\n\n".join(parts)

    async def _apply_edits(self, response: str) -> str:
        """Parse the LLM JSON response and write all listed file edits."""
        try:
            data: dict[str, Any] = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from a code-fenced block
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            cleaned = cleaned.removesuffix("```")
            try:
                data = json.loads(cleaned.strip())
            except json.JSONDecodeError as e:
                return (
                    f"Could not parse structured JSON output ({e}). "
                    f"Raw response (truncated): {response[:_MAX_SUMMARY_LEN]}"
                )

        edits = data.get("edits", [])
        summary = data.get("summary", f"Completed task: {self.task.description}")

        for edit in edits:
            path = edit.get("path", "")
            content = edit.get("content", "")
            if not path:
                continue
            await self.client.write_file(path, content)

        if not edits:
            summary = summary or "No changes were made."

        return str(summary)[:_MAX_SUMMARY_LEN]
