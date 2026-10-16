from __future__ import annotations

import re

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or Settings())

    async def run(self) -> str:
        file_contents: dict[str, str] = {}
        for path in self.task.files:
            try:
                file_contents[path] = await self.client.read_file(path)
            except FileNotFoundError:
                file_contents[path] = "(file does not exist yet)"
            except OSError:
                file_contents[path] = "(unable to read file)"

        files_section = "\n".join(
            f"### {path}\n{content}" for path, content in file_contents.items()
        ) if file_contents else "none"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Current file contents:\n{files_section}\n"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        file_blocks = re.findall(r"FILE: (.+?)\n(.*?)\nENDFILE", response, re.DOTALL)
        for path, content in file_blocks:
            await self.client.write_file(path.strip(), content)

        if file_blocks:
            summary_match = re.search(r"SUMMARY:\s*(.*)", response, re.DOTALL)
            return summary_match.group(1).strip() if summary_match else f"Wrote {len(file_blocks)} file(s)"

        return response
