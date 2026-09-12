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
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = Path(self.client.settings.workspace)
        file_contents: dict[str, str] = {}

        for path in self.task.files:
            try:
                file_contents[path] = await self.client.read_file(workspace / path)
            except Exception:
                file_contents[path] = "(file does not exist yet)"

        files_section = "".join(
            f"\n--- {path} ---\n{content}\n--- end {path} ---\n"
            for path, content in file_contents.items()
        )
        files_to_modify = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to modify: {files_to_modify}\n\n"
            f"Current file contents:{files_section}\n\n"
            "Return JSON only (no markdown, no explanation) with this exact shape:\n"
            '{\n'
            '  "files": {\n'
            '    "path/to/file.py": "complete new file content"\n'
            '  },\n'
            '  "summary": "Brief description of changes",\n'
            '  "issues": []\n'
            '}\n'
            "Include every file you create or modify in the files object, with its complete new content. "
            "Use an empty files object if no files are needed."
        )

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError, ValueError):
            return response

        if not isinstance(data, dict):
            return response

        files = data.get("files", {})
        if not isinstance(files, dict):
            return response

        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = []

        if self.client.settings.enable_file_edits:
            for path, content in files.items():
                if not isinstance(path, str) or not isinstance(content, str):
                    issues.append("The response contained an invalid file entry")
                    continue

                try:
                    await self.client.write_file(workspace / path, content)
                except Exception as error:
                    issues.append(f"Could not write {path}: {error}")

        summary = data.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = "Task completed"

        if issues:
            issue_text = "; ".join(str(issue) for issue in issues)
            summary = f"{summary}\nIssues: {issue_text}"
        return summary

