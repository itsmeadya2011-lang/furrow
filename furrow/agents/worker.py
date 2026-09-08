from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        logger.info("worker_started", task_id=self.task.id)
        files_context = ", ".join(self.task.files) if self.task.files else "any relevant files"
        conversation = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {files_context}

You have access to these tools:
- read_file: {{"path": "/path/to/file"}} -> read the contents of a file
- write_file: {{"path": "/path/to/file", "content": "full new contents"}} -> write a file
- done: no more tool calls needed, return a summary

Respond with JSON: {{"tool_calls": [...], "summary": "..."}} 
or when done: {{"summary": "what you did"}}

Always read a file before modifying it."""

        for _ in range(10):
            response = await self.client.complete(prompt=conversation, model=self.client.settings.worker_model)
            try:
                data = json.loads(response)
            except ValueError:
                logger.info("worker_completed", task_id=self.task.id, status="parse_error")
                return f"Failed to parse worker response: {response}"

            tool_calls = data.get("tool_calls", [])
            if not tool_calls:
                logger.info("worker_completed", task_id=self.task.id, status="done")
                return data.get("summary", "Done")

            conversation += f"\n\nPrevious response:\n{response}\n"
            for call in tool_calls:
                tool = call.get("tool")
                path = call.get("path")
                if tool == "read_file":
                    content = await self.client.read_file(path)
                    conversation += f"\nFile {path} contents:\n{content}\n"
                elif tool == "write_file":
                    await self.client.write_file(path, call.get("content", ""))
                    conversation += f"\nWrote to {path}\n"
                else:
                    conversation += f"\nUnknown tool: {tool}\n"
            conversation += "\nNow continue with the next step."

        logger.info("worker_completed", task_id=self.task.id, status="max_iterations")
        return "Max iterations reached"