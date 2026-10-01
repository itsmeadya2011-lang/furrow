from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.tools import ToolRegistry

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """Executes a single task via an LLM tool-use loop."""

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        tool_registry: ToolRegistry | None = None,
        max_iters: int = 10,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.tools = tool_registry or ToolRegistry(workspace=self.client.settings.workspace)
        self.max_iters = max_iters

    async def run(self) -> str:
        system = WORKER_PROMPT
        files = ", ".join(self.task.files) if self.task.files else "any"
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Task: {self.task.description}\n"
                    f"Files to touch: {files}\n"
                    "Implement the task using the available tools "
                    "(read_file, write_file, list_files, run_bash). "
                    "When finished, return a concise summary of what you changed."
                ),
            }
        ]
        for _ in range(self.max_iters):
            resp = await self.client.chat(
                tools=self.tools.tools,
                messages=messages,
                system=system,
                model=self.client.settings.worker_model,
            )
            if resp.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.text,
                        "tool_calls": [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in resp.tool_calls
                        ],
                    }
                )
                for tc in resp.tool_calls:
                    try:
                        result = await self.tools.execute(tc.name, tc.arguments)
                    except Exception as e:  # noqa: BLE001
                        result = f"Error: {e}"
                    messages.append(
                        {"role": "tool", "content": result, "tool_call_id": tc.id}
                    )
            else:
                if resp.text:
                    messages.append({"role": "assistant", "content": resp.text})
                return resp.text or "Task completed."
        return "Reached max iterations without a final answer."
