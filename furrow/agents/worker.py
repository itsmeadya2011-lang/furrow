from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """Executes a single task using tool-calling, writing files as needed."""

    MAX_ROUNDS = 3

    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"
        messages: list[dict] = [
            {"role": "system", "content": WORKER_PROMPT},
            {
                "role": "user",
                "content": f"Task: {self.task.description}\nFiles to touch: {files}\n",
            },
        ]
        for _ in range(self.MAX_ROUNDS):
            completion = await self.client.complete_with_tools(
                messages, model=self.client.settings.worker_model
            )
            if not completion.tool_calls:
                return completion.text or "No output produced."
            messages.append(
                {
                    "role": "assistant",
                    "content": completion.text,
                    "tool_calls": completion.tool_calls,
                }
            )
            for tc in completion.tool_calls:
                try:
                    result = await self.client.execute_tool(tc)
                except Exception as e:  # noqa: BLE001
                    result = f"Error executing {tc.name}: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result,
                    }
                )
        return "Reached maximum tool-call rounds without a final summary."
