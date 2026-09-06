from __future__ import annotations

import json
import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        # Read existing files so the LLM can reproduce them with modifications
        file_contents: list[str] = []
        for f in self.task.files:
            try:
                content = await self.client.read_file(f)
                file_contents.append(f"--- {f} ---\n{content}")
            except FileNotFoundError:
                file_contents.append(f"--- {f} ---\n(new file)")
            except Exception as e:
                file_contents.append(f"--- {f} ---\n(could not read: {e})")

        context = ""
        if file_contents:
            context = "\nExisting file contents:\n" + "\n".join(file_contents) + "\n"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"{context}"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        response = response.strip()

        # Strip markdown code fences if the model wrapped its JSON in them
        if response.startswith("```"):
            lines = response.splitlines()
            if lines and lines[-1].startswith("```"):
                response = "\n".join(lines[1:-1])
            elif lines:
                response = "\n".join(lines[1:])

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            log.warning("worker_json_parse_failed", error=str(e), response_preview=response[:200])
            return f"Worker could not parse LLM response as JSON. Summary: {response[:300]}"

        changes = data.get("changes", [])
        summary = data.get("summary", "No summary provided")

        # Apply file changes
        for change in changes:
            path = change.get("path")
            content = change.get("content", "")
            if path:
                await self.client.write_file(path, content)
                log.info("worker_file_written", path=path)

        return summary
