from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.util import parse_json_object

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace_files = self.client.list_files(self.client.settings.workspace)[:200]
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            f"Workspace files:\n"
            + "\n".join(workspace_files)
        )

        changed: list[str] = []
        max_rounds = self.client.settings.worker_rounds

        for _ in range(max_rounds):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)

            try:
                action = parse_json_object(response)
            except ValueError:
                prompt += "\n\nYour previous response was not valid JSON. Respond with exactly one action object."
                continue

            action_type = action.get("action")

            if action_type == "read":
                path = action.get("path")
                if not path:
                    prompt += "\n\nTool result: Missing 'path' for read action."
                    continue
                try:
                    content = await self.client.read_file(path)
                    prompt += f"\n\nTool result (file content of {path}):\n{content}"
                except Exception as e:
                    prompt += f"\n\nTool result (error reading {path}): {e}"

            elif action_type == "list":
                path = action.get("path", ".")
                try:
                    files = self.client.list_files(path)
                    prompt += f"\n\nTool result (files in {path}):\n" + "\n".join(files)
                except Exception as e:
                    prompt += f"\n\nTool result (error listing {path}): {e}"

            elif action_type == "write":
                path = action.get("path")
                content = action.get("content")
                if not path or content is None:
                    prompt += "\n\nTool result: Missing 'path' or 'content' for write action."
                    continue
                try:
                    await self.client.write_file(path, content)
                    changed.append(path)
                    prompt += f"\n\nTool result: File {path} written successfully."
                except Exception as e:
                    prompt += f"\n\nTool result (error writing {path}): {e}"

            elif action_type == "done":
                summary = action.get("summary", "Task completed")
                if changed:
                    return f"{summary} (files changed: {', '.join(changed)})"
                return summary

            else:
                prompt += (
                    "\n\nTool result: Unknown action. Valid actions are: "
                    "read, list, write, done."
                )

        summary = "Round limit reached without completion."
        if changed:
            return f"{summary} (files changed: {', '.join(changed)})"
        return summary
