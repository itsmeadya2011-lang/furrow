from __future__ import annotations

from typing import TYPE_CHECKING

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
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            "Use the available tools to read existing files, write new ones, and inspect the workspace as needed. "
            "Return a concise summary of what you changed and any issues."
        )
        system = "You are a worker agent. Use the tools provided to complete the task."

        assistant_text, tool_calls = await self.client.complete_with_tools(
            prompt,
            system=system,
            model=self.client.settings.worker_model,
        )

        tool_log: list[str] = []
        if assistant_text:
            tool_log.append(assistant_text)

        for call in tool_calls:
            result = await self.client.execute_tool(call["name"], call.get("input", {}))
            tool_log.append(f"[{call['name']}] {result}")

        return "\n".join(tool_log)
