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

    async def run(self) -> str:
        # Read existing files
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception as e:
                file_contents[file_path] = f"ERROR_READING_FILE: {e}"

        # Build comprehensive prompt
        prompt_parts = [WORKER_PROMPT]
        prompt_parts.append(f"\n\nTask: {self.task.description}")
        prompt_parts.append(f"\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}")
        prompt_parts.append("\n\nCurrent file contents:")
        for file_path, content in file_contents.items():
            prompt_parts.append(f"\n=== {file_path} ===\n{content}\n=== END {file_path} ===")

        prompt = "\n".join(prompt_parts)

        try:
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            # Try to parse as JSON
            # Handle potential markdown code fences
            response_clean = response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()

            data = json.loads(response_clean)
            changes = data.get("changes", [])
            summary = data.get("summary", "No summary provided")

            # Apply changes
            for change in changes:
                action = change.get("action")
                path = change.get("path")
                content = change.get("content", "")
                if action == "create" or action == "modify":
                    await self.client.write_file(path, content)
                elif action == "delete":
                    try:
                        import os

                        os.remove(path)
                    except Exception:
                        pass

            return summary
        except json.JSONDecodeError as e:
            return f"ERROR: Failed to parse JSON response. Error: {e}\n\nRaw response:\n{response}"
        except Exception as e:
            return f"ERROR: {e}"
