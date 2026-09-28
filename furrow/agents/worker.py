from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, FILE_TOOLS
from furrow.config import Provider, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files_str = ", ".join(self.task.files) if self.task.files else "any relevant files"

        project_files: list[str] = []
        try:
            project_files = self.client.list_files(self.client.settings.workspace)
        except Exception:
            pass
        project_files_str = "\n".join(project_files[:20]) if project_files else "(none)"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n"
            f"Project files:\n{project_files_str}\n"
        )

        # Use tool calling for Anthropic (supports read_file, write_file, list_files)
        if self.client.settings.provider == Provider.ANTHROPIC:
            try:
                return await self.client.complete_with_tools(
                    prompt,
                    tools=FILE_TOOLS,
                    model=self.client.settings.worker_model,
                )
            except Exception:
                # Fall back to basic completion if tool calling fails
                prompt = f"{prompt}\n\nNote: Tool calling unavailable, describe changes as text.\n"

        # Fallback for non-Anthropic providers or tool calling failures
        return await self.client.complete(prompt, model=self.client.settings.worker_model)
