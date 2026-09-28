from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.tools import TOOL_SCHEMAS, ToolRunner

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger("furrow")


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_turns: int = 20,
    ) -> None:
        self.task = task
        if client is not None:
            self.client = client
        elif settings is not None:
            self.client = LLMClient(settings=settings)
        else:
            self.client = LLMClient()
        self.settings = self.client.settings
        self.tool_runner = ToolRunner(workspace=self.settings.workspace)
        self.max_turns = max_turns
        self._available_tools = self.tool_runner.get_tool_names()

    async def run(self) -> str:
        task_info = (
            f"Task ID: {self.task.id}\n"
            f"Task description: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any relevant files'}\n"
            f"Dependencies: {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}\n"
        )
        prompt = (
            f"{WORKER_PROMPT}\n\n{task_info}\nAvailable tools: {', '.join(self._available_tools)}"
        )

        messages: list[dict] = [{"role": "user", "content": prompt}]

        logger.info("worker_start", task_id=self.task.id, max_turns=self.max_turns)

        for turn in range(self.max_turns):
            result = await self.client.complete_with_tools(
                messages,
                TOOL_SCHEMAS,
                system=WORKER_PROMPT,
                model=self.client.settings.worker_model,
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": result.text,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "args": tc.args} for tc in result.tool_calls
                    ],
                }
            )

            if not result.tool_calls:
                summary = (result.text or "").strip()
                logger.info("worker_done", task_id=self.task.id, turns=turn + 1)
                return summary or "Completed task."

            for tc in result.tool_calls:
                outcome = await self.tool_runner.execute(tc)
                messages.append(
                    {
                        "role": "tool",
                        "content": outcome,
                        "tool_call_id": tc.id,
                    }
                )

        final = await self.client.complete_with_tools(
            messages,
            TOOL_SCHEMAS,
            system=WORKER_PROMPT,
            model=self.client.settings.worker_model,
        )
        logger.warning("worker_max_turns_reached", task_id=self.task.id, max_turns=self.max_turns)
        return (
            final.text or ""
        ).strip() or f"Reached max turns ({self.max_turns}). Task may be incomplete."
