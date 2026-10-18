from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient, extract_json

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

    async def run(self) -> str:
        # Read the current contents of each file to touch (if it exists).
        file_contents: dict[str, str | None] = {}
        for file_path in self.task.files:
            try:
                file_contents[file_path] = await self.client.read_file(file_path)
            except FileNotFoundError:
                file_contents[file_path] = None

        files_list = ", ".join(self.task.files) if self.task.files else "any"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_list}\n\n"
            "Current file contents:\n"
        )
        for path, content in file_contents.items():
            marker = "(does not exist yet)" if content is None else content
            prompt += f"\n--- {path} ---\n{marker}\n"
        prompt += (
            "\nReturn only the JSON object mapping each file path to its complete new content."
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        new_contents = extract_json(response)
        if not isinstance(new_contents, dict):
            raise ValueError(f"Worker returned non-object JSON: {response}")

        written: list[str] = []
        for path, content in new_contents.items():
            await self.client.write_file(path, content)
            written.append(path)

        return f"Wrote {len(written)} file(s): {', '.join(written)}"
