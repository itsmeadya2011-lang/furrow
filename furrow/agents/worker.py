from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger(__name__)


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        log.info("Task started", task_id=self.task.id, description=self.task.description)
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        parsed = self._parse_worker_response(response)
        for path, content in parsed["file_blocks"].items():
            try:
                await self.client.write_file(path, content)
                log.info("File written", task_id=self.task.id, path=path)
            except OSError as e:
                log.error("Failed to write file", task_id=self.task.id, path=path, error=str(e))
        log.info("Task completed", task_id=self.task.id, files_changed=len(parsed["files_changed"]))
        return parsed["summary"]

    @staticmethod
    def _parse_worker_response(response: str) -> dict[str, Any]:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else ""
            text = text.removesuffix("```")

        summary_match = re.search(r"SUMMARY:\s*(.*?)(?=FILES_CHANGED:|$)", text, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""

        files_changed: list[str] = []
        files_match = re.search(r"FILES_CHANGED:\s*(.*?)(?=FILE:\s|\Z)", text, re.DOTALL)
        if files_match:
            for line in files_match.group(1).splitlines():
                line = line.strip()
                if line.startswith("- "):
                    path = line[2:].strip()
                    if path:
                        files_changed.append(path)

        file_blocks: dict[str, str] = {}
        for match in re.finditer(
            r"FILE:\s*(\S+)\s*\nCONTENT:\s*\n(.*?)\n?ENDFILE", text, re.DOTALL
        ):
            path = match.group(1).strip()
            if path:
                file_blocks[path] = _strip_code_fences(match.group(2))

        return {"summary": summary, "files_changed": files_changed, "file_blocks": file_blocks}


def _strip_code_fences(content: str) -> str:
    stripped = content.strip("\n")
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1 :] if first_newline != -1 else ""
        stripped = stripped.removesuffix("```")
    return stripped.strip("\n")
