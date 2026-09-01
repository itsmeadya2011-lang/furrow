from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path(settings.workspace) if settings else Path.cwd()

    async def run(self) -> str:
        """Execute the task by reading files, generating changes, and writing results."""
        file_context = await self._gather_file_context()

        prompt = f"""{WORKER_PROMPT}

## Task
{self.task.description}

## Files to Touch
{', '.join(self.task.files) if self.task.files else 'Determine based on task'}

## File Context
{file_context}

## Instructions
1. Analyze the task and file context
2. Determine what changes need to be made
3. Return a JSON response with your changes:

```json
{{
  "changes": [
    {{
      "file": "path/to/file.py",
      "action": "create|modify|delete",
      "content": "full file content for create/modify, empty for delete"
    }}
  ],
  "summary": "Brief description of what was done",
  "issues": ["any issues encountered"]
}}
```

If no file changes are needed, return empty changes array with explanation in summary."""

        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        try:
            data = json.loads(response)
            return await self._apply_changes(data)
        except json.JSONDecodeError:
            # If LLM didn't return JSON, treat the response as a summary
            return f"Task analysis complete (no file changes): {response[:500]}"

    async def _gather_file_context(self) -> str:
        """Read relevant files to provide context for the task."""
        context_parts = []

        # If specific files are mentioned, read them
        if self.task.files:
            for file_path in self.task.files:
                full_path = self.workspace / file_path
                if full_path.exists():
                    try:
                        content = await self.client.read_file(full_path)
                        context_parts.append(f"### {file_path}\n```\n{content}\n```\n")
                    except Exception as e:
                        context_parts.append(f"### {file_path}\n[Could not read: {e}]\n")
                else:
                    context_parts.append(f"### {file_path}\n[File does not exist - needs to be created]\n")

        # If no specific files, list workspace structure
        if not context_parts:
            files = self.client.list_files(self.workspace)
            if files:
                context_parts.append("### Workspace Structure\n")
                # Limit to first 50 files to avoid context overflow
                for f in files[:50]:
                    context_parts.append(f"- {f}")
                if len(files) > 50:
                    context_parts.append(f"\n... and {len(files) - 50} more files")

        return "\n".join(context_parts) if context_parts else "[No file context available]"

    async def _apply_changes(self, data: dict) -> str:
        """Apply the changes specified by the LLM response."""
        changes = data.get("changes", [])
        summary = data.get("summary", "No summary provided")
        issues = data.get("issues", [])

        if not changes:
            return f"No file changes made. {summary}"

        applied = []
        errors = []

        for change in changes:
            file_path = change.get("file", "")
            action = change.get("action", "modify")
            content = change.get("content", "")

            if not file_path:
                errors.append("Change missing file path")
                continue

            full_path = self.workspace / file_path

            try:
                if action == "create" or action == "modify":
                    await self.client.write_file(full_path, content)
                    applied.append(f"{action}: {file_path}")
                elif action == "delete":
                    if full_path.exists():
                        full_path.unlink()
                        applied.append(f"delete: {file_path}")
                    else:
                        errors.append(f"Cannot delete {file_path}: file does not exist")
                else:
                    errors.append(f"Unknown action '{action}' for {file_path}")
            except Exception as e:
                errors.append(f"Failed to {action} {file_path}: {e}")

        result_parts = [f"Summary: {summary}"]
        if applied:
            result_parts.append(f"Applied: {', '.join(applied)}")
        if errors:
            result_parts.append(f"Errors: {'; '.join(errors)}")
        if issues:
            result_parts.append(f"Issues: {'; '.join(issues)}")

        return "\n".join(result_parts)
