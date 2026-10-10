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
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n"
            "When you need to write a file, respond with a JSON object in a ```tool``` code block:\n"
            '{"tool": "write_file", "path": "src/example.py", "content": "..."}\n'
            "You may emit multiple tool calls. End with a final summary line prefixed with 'RESULT: '."
        )
        written: list[str] = []
        for _ in range(5):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            for match in re.finditer(r"```tool\s*\n?(.*?)\n?```", response, re.DOTALL):
                try:
                    call = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                if call.get("tool") == "write_file":
                    await self.client.write_file(call["path"], call.get("content", ""))
                    written.append(call["path"])
            if response.strip().startswith("RESULT:"):
                summary = response.strip()[len("RESULT:") :].strip()
                return f"Wrote {len(written)} file(s): {', '.join(written)}. {summary}"
            prompt = (
                f"{response}\n\n"
                "If you have finished, reply with a line starting with 'RESULT: ' and a summary. "
                "Otherwise emit more ```tool``` blocks."
            )
        return f"Wrote {len(written)} file(s): {', '.join(written)}. (max iterations reached)"
