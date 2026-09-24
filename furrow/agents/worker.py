from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class FileOperation:
    def __init__(self, op: str, path: str, content: str = "") -> None:
        self.op = op  # create, edit, delete
        self.path = path
        self.content = content


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self) -> str:
        # Read context files first
        context = await self._read_context_files()
        
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nContext files:\n{context}\n\nReturn JSON only with this shape:\n{{\n  \"operations\": [\n    {{\"op\": \"create\", \"path\": \"file.py\", \"content\": \"...\"}},\n    {{\"op\": \"edit\", \"path\": \"file.py\", \"content\": \"...\"}},\n    {{\"op\": \"delete\", \"path\": \"file.py\"}}\n  ],\n  \"summary\": \"Brief summary of changes\"\n}}"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        return await self._apply_operations(response)

    async def _read_context_files(self) -> str:
        """Read files mentioned in the task to provide context."""
        if not self.task.files:
            return "No specific files provided."
        
        parts = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                parts.append(f"=== {file_path} ===\n{content}")
            except FileNotFoundError:
                parts.append(f"=== {file_path} === (not found)")
            except Exception as e:
                parts.append(f"=== {file_path} === (error: {e})")
        return "\n\n".join(parts)

    async def _apply_operations(self, response: str) -> str:
        """Parse LLM response and apply file operations."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            operations = data.get("operations", [])
            summary = data.get("summary", "Task completed")
            
            applied = []
            for op_data in operations:
                op = FileOperation(
                    op=op_data.get("op", ""),
                    path=op_data.get("path", ""),
                    content=op_data.get("content", "")
                )
                result = await self._apply_operation(op)
                applied.append(result)
            
            return f"{summary}\n\nApplied operations:\n" + "\n".join(applied)
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: just return the raw response if parsing fails
            return f"Task completed (raw response): {response}"

    async def _apply_operation(self, op: FileOperation) -> str:
        """Apply a single file operation."""
        path = Path(op.path)
        
        if op.op == "create":
            await self.client.write_file(path, op.content)
            return f"Created: {path}"
        elif op.op == "edit":
            # For edit, we replace the entire file content
            await self.client.write_file(path, op.content)
            return f"Edited: {path}"
        elif op.op == "delete":
            if path.exists():
                path.unlink()
                return f"Deleted: {path}"
            return f"Delete skipped (not found): {path}"
        else:
            return f"Unknown operation: {op.op} on {path}"
