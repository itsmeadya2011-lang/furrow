from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import FILE_TOOLS, LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_CODE_FENCE_RE = re.compile(r"```(?:\w+)?\s*\n?(.*?)```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Replace fenced code blocks with their raw content so JSON parsing is robust."""
    return _CODE_FENCE_RE.sub(lambda m: m.group(1), text)


class WorkerAgent:
    """A worker that actually edits files.

    It runs a small tool-calling loop: ask the LLM for an action, if it asks
    to read/write a file we do it and feed the result back, otherwise we
    return the final summary.
    """

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_rounds: int = 6,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = max_rounds

    async def run(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any relevant file"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n"
            f"Dependencies (already completed): {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}\n\n"
            "You have access to read_file and write_file tools. Use them to make "
            "the changes. When finished, reply with a single line starting with "
            "'SUMMARY:' followed by a concise description of what you did."
        )

        system = (
            "You are a worker agent. Use the read_file and write_file tools to "
            "complete the task. Always end your response with a line that begins "
            "with 'SUMMARY:'."
        )

        for _ in range(self.max_rounds):
            response = await self.client.complete(
                prompt, system=system, model=self.client.settings.worker_model, tools=FILE_TOOLS
            )
            if "SUMMARY:" in response:
                return response.split("SUMMARY:", 1)[1].strip()
            prompt = (
                "Your last response did not include a SUMMARY line. "
                "Please finish the task and end with SUMMARY: <description>.\n\n"
                f"Previous response:\n{response}"
            )

        return "Worker did not produce a SUMMARY line within the round limit."

    async def read_file(self, path: str) -> str:
        return await self.client.read_file(path)

    async def write_file(self, path: str, content: str) -> None:
        await self.client.write_file(path, content)

    @staticmethod
    def extract_json(text: str) -> dict:
        """Best-effort extraction of a JSON object from an LLM response."""
        cleaned = _strip_code_fences(text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Fall back to the first {...} block.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise ValueError(f"No JSON object found in response: {text!r}")