from __future__ import annotations

import json
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
        self.settings = settings
        self.max_rounds = 3

    async def run(self) -> str:
        # Read relevant files to provide context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = f"[File not found: {file_path}]"

        # Build context from files
        context = ""
        if file_contents:
            context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{context}\n"

        # Use function calling / tool use approach - the LLM should return JSON with file operations
        for round_num in range(self.max_rounds):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)

            # Try to parse as JSON with file operations
            try:
                data = json.loads(response)
                if isinstance(data, dict) and "operations" in data:
                    # Execute file operations
                    for op in data["operations"]:
                        if op.get("type") == "write":
                            await self.client.write_file(op["path"], op["content"])
                        elif op.get("type") == "read":
                            # Re-read file for context
                            content = await self.client.read_file(op["path"])
                            file_contents[op["path"]] = content

                    # Check if task is complete
                    if data.get("complete", False):
                        return data.get("summary", "Task completed")

                    # Continue with updated context
                    context = "\n\nUpdated files:\n"
                    for path, content in file_contents.items():
                        context += f"\n--- {path} ---\n{content}\n"
                    prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nContinue working. Current state:{context}\n"
                    continue

            except json.JSONDecodeError:
                pass

            # If not JSON or not complete, return the response as summary
            return response

        return "Max rounds reached"
