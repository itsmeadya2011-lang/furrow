from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

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
        files_to_touch = self.task.files or []

        file_contents: list[tuple[str, str]] = []
        for path in files_to_touch:
            try:
                content = await self.client.read_file(path)
                file_contents.append((path, content))
            except Exception:
                file_contents.append((path, "<NEW FILE>"))

        contents_str = "\n\n".join(
            f"=== {path} ===\n{content}" for path, content in file_contents
        )

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(files_to_touch) if files_to_touch else 'any'}\n\n"
            f"FILE CONTENTS:\n{contents_str}\n\n"
            "Return a single JSON object with shape {\"files\": [{\"path\": str, \"content\": str}], \"summary\": str}. "
            "No markdown, no explanation outside the JSON."
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        parsed: dict[str, Any]
        try:
            parsed = json.loads(response)
        except Exception:
            strict_prompt = prompt + "\n\nSTRICT: Return ONLY the JSON object, no markdown fences, no prose."
            response = await self.client.complete(strict_prompt, model=self.client.settings.worker_model)
            try:
                parsed = json.loads(response)
            except Exception as e:
                raise ValueError(f"Failed to parse worker response as JSON: {e}\nResponse: {response}") from e

        files = parsed.get("files", [])
        if not isinstance(files, list):
            files = []

        for file_info in files:
            if not isinstance(file_info, dict):
                continue
            path = file_info.get("path")
            content = file_info.get("content", "")
            if path is None:
                continue
            await self.client.write_file(path, content)

        summary = parsed.get("summary", "")
        if not isinstance(summary, str):
            summary = str(summary) if summary is not None else ""

        return summary
