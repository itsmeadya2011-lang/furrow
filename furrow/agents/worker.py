from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient, WORKER_TOOLS

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = self.client.settings.workspace

    async def run(self) -> str:
        # Build context about the workspace
        files_context = ""
        if self.task.files:
            files_context = "\nRelevant files:\n"
            for f in self.task.files:
                try:
                    content = await self.client.read_file(self.workspace / f)
                    files_context += f"\n--- {f} ---\n{content}\n"
                except FileNotFoundError:
                    files_context += f"\n--- {f} ---\n(File not found)\n"

        # List workspace files for context
        workspace_files = self.client.list_files(self.workspace)
        files_list = "\n".join(workspace_files[:50]) if workspace_files else "(empty)"

        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}

Workspace files:
{files_list}
{files_context}

Your goal: Implement the task completely. Use the read_file, write_file, and list_files tools as needed.
Return a concise summary of what you changed."""

        result = await self.client.complete_with_tools(
            prompt=prompt,
            model=self.client.settings.worker_model,
            tools=WORKER_TOOLS,
            max_turns=5,
        )
        return result