from __future__ import annotations

import json
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
        workspace = self.client.settings.workspace

        # List the files present in the workspace
        workspace_files = self.client.list_files(workspace)

        # Build file context: read each file the task wants to touch
        files_context_parts: list[str] = []
        if workspace_files:
            files_context_parts.append(
                "Files in workspace:\n" + "\n".join(f"  - {f}" for f in workspace_files)
            )
        else:
            files_context_parts.append("Files in workspace: (none)")

        files_context_parts.append("\nContents of files to touch:")
        for path in self.task.files:
            try:
                content = await self.client.read_file(path)
                files_context_parts.append(f"\n--- {path} ---\n{content}")
            except FileNotFoundError:
                files_context_parts.append(f"\n--- {path} (does not exist yet) ---")

        files_context = "\n".join(files_context_parts)

        # Build the prompt from the template, filling in runtime placeholders
        prompt = WORKER_PROMPT
        prompt = prompt.replace("{workspace}", str(workspace))
        prompt = prompt.replace("{files_context}", files_context)
        prompt = prompt.replace("{task_description}", self.task.description)
        prompt = prompt.replace(
            "{files_to_touch}", ", ".join(self.task.files) if self.task.files else "any"
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Parse the structured JSON response returned by the LLM
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            log_path = Path(workspace) / "furrow_logs" / f"worker_task_{self.task.id}.md"
            await self.client.write_file(str(log_path), response)
            return (
                f"Failed to parse worker output as JSON for task {self.task.id}. "
                f"Raw response written to {log_path}."
            )

        files = data.get("files", [])
        summary = data.get("summary", "")

        # Write each file returned by the LLM
        written: list[str] = []
        for entry in files:
            try:
                path = entry["path"]
                content = entry["content"]
            except (KeyError, TypeError):
                continue
            await self.client.write_file(path, content)
            written.append(path)

        if summary:
            return summary
        return f"Wrote {len(written)} files: " + ", ".join(written)
