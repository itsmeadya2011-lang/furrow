from __future__ import annotations

import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

_CODE_BLOCK_RE = re.compile(r"```(?P<info>[^\n]*)\n(?P<code>.*?)```", re.DOTALL)
_PATH_LIKE_RE = re.compile(r"^[\w.-]+\.\w+$")


def _path_from_info(info: str) -> str | None:
    for token in info.split():
        if "/" in token or _PATH_LIKE_RE.match(token):
            return token
    return None


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        written: list[str] = []
        files = self.task.files
        for index, match in enumerate(_CODE_BLOCK_RE.finditer(response)):
            path = _path_from_info(match.group("info"))
            if path is None:
                path = files[index] if index < len(files) else None
            if path is None:
                path = f"task_{self.task.id}_{index}.py"
            await self.client.write_file(path, match.group("code"))
            written.append(path)

        if not written:
            return "No code blocks found in response; no files written."
        return f"Wrote {len(written)} file(s): {', '.join(written)}"
