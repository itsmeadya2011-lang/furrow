from __future__ import annotations

import asyncio
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
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path(settings.workspace) if settings else Path.cwd()

    async def run(self) -> str:
        prompt = self._build_prompt()
        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )
        await self._apply_changes(response)
        return self._summarize(response)

    def _build_prompt(self) -> str:
        files_str = ", ".join(self.task.files) if self.task.files else "any relevant files"
        context = ""
        for f in self.task.files:
            p = self.workspace / f
            if p.exists():
                try:
                    content = p.read_text()
                    context += f"\n\n--- Current content of {f} ---\n{content}\n--- end ---"
                except Exception:
                    pass
        return (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n"
            f"Workspace root: {self.workspace}\n"
            f"{context}\n\n"
            "Return your changes as a JSON object with this shape:\n"
            '{"files": [{"path": "relative/path", "action": "create|update|delete", "content": "..."}]}\n'
            "Only include files you actually changed. No markdown, no explanation outside the JSON."
        )

    async def _apply_changes(self, response: str) -> None:
        data = self._parse_response(response)
        if not data or "files" not in data:
            return
        for file_entry in data["files"]:
            path = self.workspace / file_entry["path"]
            action = file_entry.get("action", "update")
            content = file_entry.get("content", "")
            if action == "delete":
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)

    def _parse_response(self, response: str) -> dict | None:
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        match = re.search(r'\{[\s\S]*"files"[\s\S]*\}', response)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _summarize(self, response: str) -> str:
        data = self._parse_response(response)
        if data and "files" in data:
            changed = [f["path"] for f in data["files"]]
            return f"Changed {len(changed)} file(s): {', '.join(changed)}"
        return f"Task completed. Response: {response[:200]}"