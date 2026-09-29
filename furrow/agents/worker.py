from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": WORKER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {self.task.description}\n"
                    f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
                ),
            },
        ]

        max_rounds = 10
        for _ in range(max_rounds):
            response = await self.client.complete_with_tools(
                messages=messages,
                model=self.client.settings.worker_model,
            )

            if not response.tool_calls:
                return response.text or ""

            messages.append(
                {
                    "role": "assistant",
                    "content": response.text or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in response.tool_calls
                    ],
                }
            )

            for tc in response.tool_calls:
                try:
                    result = await self.client.execute_tool(tc.name, tc.arguments)
                except Exception as e:
                    result = f"Error executing tool {tc.name}: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        return "Task may be incomplete — maximum rounds reached."
