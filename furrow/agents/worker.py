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
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        context = await self._gather_context()
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nContext:\n{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        result = self._parse_response(response)
        for change in result.get("changes", []):
            path = self._resolve_path(change["path"])
            await self.client.write_file(path, change["content"])
        return result.get("summary", "")

    async def _gather_context(self) -> str:
        workspace = self.client.settings.workspace
        if self.task.files:
            sections = []
            for rel_path in self.task.files:
                path = self._resolve_path(rel_path)
                try:
                    content = await self.client.read_file(path)
                except OSError:
                    content = "(file does not exist yet)"
                sections.append(f"--- {rel_path} ---\n{content}")
            return "\n\n".join(sections)
        files = self.client.list_files(workspace)
        if not files:
            return "(empty workspace)"
        return "Files in workspace:\n" + "\n".join(files)

    def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.client.settings.workspace / p

    @staticmethod
    def _parse_response(response: str) -> dict:
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Failed to parse changes from LLM: no JSON object\nResponse: {response}")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse changes from LLM: {e}\nResponse: {response}")
