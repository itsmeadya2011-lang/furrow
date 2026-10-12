from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


# Matches a fenced code block with an explicit filename header, e.g.:
#   FILE: path/to/file.py
#   ```python
#   ...content...
#   ```
_FILE_BLOCK_RE = re.compile(
    r"FILE:\s*(?P<path>[^\n]+?)\s*\n```[^\n]*\n(?P<content>.*?)```",
    re.DOTALL,
)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            "When you make changes, output them using this exact format so they can be applied "
            "automatically:\n\n"
            "FILE: path/to/file.py\n"
            "```python\n"
            "# full new file contents (or full file for new files)\n"
            "```\n\n"
            "For edits to existing files, prefer outputting the FULL rewritten file contents "
            "under a FILE: header. Output only the file blocks and a one-line summary at the end."
        )
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        await self._apply_changes(response)
        summary = self._extract_summary(response)
        return summary

    async def _apply_changes(self, response: str) -> None:
        """Parse FILE: blocks from the LLM response and write them to disk."""
        matches = _FILE_BLOCK_RE.findall(response)
        if not matches:
            return
        for path, content in matches:
            path = path.strip()
            # Normalize line endings: strip trailing CR and convert CRLF -> LF.
            content = content.removesuffix("\r").replace("\r\n", "\n")
            await self.client.write_file(path, content)

    @staticmethod
    def _extract_summary(response: str) -> str:
        """Return the trailing non-FILE line as a summary, if any."""
        lines = [ln for ln in response.splitlines() if not ln.strip().startswith("FILE:")]
        summary = ""
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("```"):
                summary = stripped
                break
        if not summary:
            return "No summary provided."
        # Strip an optional "SUMMARY:" prefix.
        prefix = "SUMMARY:"
        if summary.upper().startswith(prefix):
            summary = summary[len(prefix):].strip()
        return summary