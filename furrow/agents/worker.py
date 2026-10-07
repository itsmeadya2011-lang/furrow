from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 3

    async def run(self) -> str:
        # Read relevant files first
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception:
                file_contents[file_path] = f"[File not found: {file_path}]"

        file_context = ""
        if file_contents:
            file_context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                file_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}{file_context}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"

        for round_num in range(self.max_rounds):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)

            # Try to parse structured output for file edits
            edits = self._parse_edits(response)
            if edits:
                applied = []
                for edit in edits:
                    try:
                        await self._apply_edit(edit)
                        applied.append(edit["path"])
                    except Exception as e:
                        return f"Failed to apply edit to {edit['path']}: {e}"

                if applied:
                    # Re-read files to get updated context for next round
                    file_contents = {}
                    for file_path in self.task.files:
                        try:
                            content = await self.client.read_file(file_path)
                            file_contents[file_path] = content
                        except Exception:
                            file_contents[file_path] = f"[File not found: {file_path}]"

                    file_context = "\n\nUpdated files:\n"
                    for path, content in file_contents.items():
                        file_context += f"\n--- {path} ---\n{content}\n"

                    prompt = (
                        f"Files updated: {', '.join(applied)}.\n"
                        f"Continue with the task if needed, or provide final summary.\n{file_context}"
                    )
                    continue

            # No edits found, return the response as summary
            return response.strip()

        return "Max rounds reached. " + response.strip()

    def _parse_edits(self, response: str) -> list[dict]:
        """Parse structured edits from LLM response.
        
        Expected format:
        ```edit
        path: path/to/file.py
        <<<<<<< ORIGINAL
        original content
        =======
        new content
        >>>>>>> UPDATED
        ```
        """
        edits = []
        pattern = r"```edit\npath: (.+?)\n<<<<<<< ORIGINAL\n(.*?)\n=======\n(.*?)\n>>>>>>> UPDATED\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            path, original, updated = match
            edits.append({"path": path.strip(), "original": original, "updated": updated})
        return edits

    async def _apply_edit(self, edit: dict) -> None:
        path = Path(edit["path"])
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            content = await self.client.read_file(path)
            if edit["original"] in content:
                content = content.replace(edit["original"], edit["updated"])
            else:
                # Try to find the original with some flexibility
                raise ValueError(f"Original content not found in {path}")
        else:
            content = edit["updated"]

        await self.client.write_file(path, content)
