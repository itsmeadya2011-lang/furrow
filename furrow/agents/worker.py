from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, WORKER_TOOL_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.tools import execute_tool, get_tool_schemas

if TYPE_CHECKING:
    pass


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_rounds: int = 5,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()
        self.max_rounds = max_rounds

    async def run(self) -> str:
        """Run the worker agent with tool use to complete the task."""
        system_prompt = f"{WORKER_PROMPT}\n\n{WORKER_TOOL_PROMPT}"

        # Build initial prompt with task details
        files_context = ""
        if self.task.files:
            files_context = f"\nFiles to touch: {', '.join(self.task.files)}"

        prompt = (
            f"Task: {self.task.description}{files_context}\n\n"
            "Begin by exploring the codebase to understand the task. "
            "Use tools to read files, then implement the changes. "
            "Return a summary when done."
        )

        messages: list[dict] = [
            {"role": "user", "content": prompt},
        ]

        tool_schemas = get_tool_schemas()

        for round_num in range(self.max_rounds):
            response = await self.client.complete_with_tools(
                messages=messages,
                system=system_prompt,
                tools=tool_schemas,
                model=self.client.settings.worker_model,
            )

            if response.get("tool_calls"):
                # Execute tool calls
                messages.append({"role": "assistant", "content": response.get("content", ""), "tool_calls": response["tool_calls"]})

                for tool_call in response["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])

                    result = await execute_tool(tool_name, **tool_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result.to_dict()),
                    })
            else:
                # No tool calls, task is complete
                return response.get("content", "Task completed (no output)")

        return f"Task completed after {self.max_rounds} rounds (max rounds reached)"