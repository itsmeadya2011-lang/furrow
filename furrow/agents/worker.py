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
        prompt = self._build_prompt()
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._execute_operations(response)

    def _build_prompt(self) -> str:
        files_hint = ', '.join(self.task.files) if self.task.files else 'any files needed'
        return f"""{WORKER_PROMPT}

Task ID: {self.task.id}
Task: {self.task.description}
Files to touch: {files_hint}

IMPORTANT: You MUST respond with a JSON object containing file operations.
Format:
{{
  "summary": "Brief description of what you implemented",
  "operations": [
    {{"action": "write", "path": "path/to/file", "content": "file contents here"}},
    {{"action": "edit", "path": "path/to/file", "old_text": "text to replace", "new_text": "replacement text"}},
    {{"action": "create_directory", "path": "path/to/dir"}}
  ]
}}

Valid actions: "write" (create/overwrite file), "edit" (replace text in existing file), "create_directory"
Include ALL file contents for "write" operations - partial writes will overwrite the entire file.
"""

    async def _execute_operations(self, response: str) -> str:
        """Parse LLM response and execute file operations."""
        operations = self._parse_operations(response)
        if not operations:
            return f"No valid operations found in response. Raw response: {response[:500]}"

        results = []
        for op in operations:
            try:
                result = await self._execute_single_operation(op)
                results.append(result)
            except Exception as e:
                results.append(f"FAILED {op.get('action', 'unknown')} {op.get('path', 'unknown')}: {e}")

        summary = operations[0].get('summary', 'Operations completed') if operations else 'Operations completed'
        return f"{summary}\n" + "\n".join(results)

    def _parse_operations(self, response: str) -> list[dict]:
        """Extract operations from LLM response."""
        # Try to find JSON object in response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        if not isinstance(data, dict):
            return []

        operations = data.get('operations', [])
        if not isinstance(operations, list):
            return []

        # Add summary to each operation for reference
        summary = data.get('summary', '')
        for op in operations:
            if isinstance(op, dict):
                op['summary'] = summary

        return operations

    async def _execute_single_operation(self, op: dict) -> str:
        """Execute a single file operation."""
        action = op.get('action')
        path = op.get('path')

        if not action or not path:
            return f"SKIP: Missing action or path in operation: {op}"

        if action == 'write':
            content = op.get('content', '')
            await self.client.write_file(path, content)
            return f"WRITTEN: {path} ({len(content)} chars)"

        elif action == 'edit':
            old_text = op.get('old_text', '')
            new_text = op.get('new_text', '')
            try:
                existing = await self.client.read_file(path)
                if old_text not in existing:
                    return f"SKIP: Could not find text to replace in {path}"
                updated = existing.replace(old_text, new_text, 1)
                await self.client.write_file(path, updated)
                return f"EDITED: {path}"
            except FileNotFoundError:
                return f"SKIP: File not found for edit: {path}"

        elif action == 'create_directory':
            from pathlib import Path
            Path(path).mkdir(parents=True, exist_ok=True)
            return f"DIRECTORY CREATED: {path}"

        else:
            return f"SKIP: Unknown action '{action}'"
