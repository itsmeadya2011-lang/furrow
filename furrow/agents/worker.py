from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
        written_files: list[str] = []

        for block in code_blocks:
            filename_match = re.search(r"(?:#|//)\s*File:\s*(.+)", block)
            if not filename_match:
                continue
            filepath = filename_match.group(1).strip()
            content = re.sub(rf"(?:#|//)\s*File:\s*{re.escape(filepath)}\n?", "", block, count=1).strip()
            target = Path(filepath)
            if not target.is_absolute():
                target = Path(self.client.settings.workspace) / target
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written_files.append(str(target))
            logger.info("Wrote file: %s", target)

        if written_files:
            summary_match = re.search(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
            summary_text = re.sub(r"```(?:\w+)?\n|```", "", response).strip() if summary_match else response.strip()
            return f"Implemented task: wrote files {written_files}. Summary: {summary_text}"

        return response
