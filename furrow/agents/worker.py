from __future__ import annotations

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
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Parse file blocks: ```file:path\ncontent```
        file_pattern = re.compile(r"```file:(.+?)\n([\s\S]*?)```")
        matches = file_pattern.findall(response)
        
        for path, content in matches:
            path = path.strip()
            await self.client.write_file(path, content)
        
        # Extract summary
        summary_match = re.search(r"SUMMARY:\s*(.+)", response)
        if summary_match:
            return summary_match.group(1).strip()
        
        if matches:
            return f"Edited {len(matches)} file(s): {', '.join(p.strip() for p, _ in matches)}"
        return "Task completed (no file changes detected in response)"
