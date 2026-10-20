from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


def _extract_json_object(response: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, tolerating surrounding text."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not extract JSON object from response: {response!r}")


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
        # Gather context from the files this task touches so the LLM can edit
        # them accurately instead of generating blindly.
        context_blocks: list[str] = []
        for path in self.task.files:
            try:
                content = await self.client.read_file(path)
            except FileNotFoundError:
                content = "(file does not exist yet)"
            context_blocks.append(f"### {path}\n{content}")
        context = "\n\n".join(context_blocks) if context_blocks else "(no specific files)"

        targets = ", ".join(self.task.files) if self.task.files else "any relevant files"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {targets}\n"
            f"Current file contents:\n{context}\n\n"
            "Return ONLY a JSON object mapping each file path you create or modify "
            "to its complete new file contents. Do not wrap in markdown fences."
        )
        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        edits = _extract_json_object(response)

        written: list[str] = []
        for path, content in edits.items():
            if not isinstance(content, str):
                continue
            await self.client.write_file(path, content)
            written.append(path)

        if written:
            return f"Wrote {len(written)} file(s): {', '.join(written)}"
        return "No files written."
