from __future__ import annotations

import json
import re
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
        workspace = self.client.settings.workspace
        files_to_touch = self.task.files or []

        # Build the prompt with current file contents.
        sections = [f"Task: {self.task.description}"]
        if files_to_touch:
            sections.append(f"Files to touch: {', '.join(files_to_touch)}")

        for path in files_to_touch:
            try:
                content = await self.client.read_file(workspace / path)
            except (FileNotFoundError, IsADirectoryError, OSError):
                content = "<file does not exist yet>"
            sections.append(f"--- Current content of {path} ---\n{content}\n--- end ---")

        prompt = f"{WORKER_PROMPT}\n\n" + "\n\n".join(sections)

        raw = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse the JSON response, tolerating markdown code fences.
        data = self._parse_json(raw)
        if data is None:
            return raw

        edits = data.get("edits", [])
        for edit in edits:
            path = edit.get("path")
            content = edit.get("content")
            if path and content is not None:
                await self.client.write_file(workspace / path, content)

        return data.get("summary", raw)

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        """Try to extract a JSON object from the raw LLM response."""
        raw = raw.strip()

        # Try direct parse first.
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code fences.
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        # Try to find the first { ... } block.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        return None