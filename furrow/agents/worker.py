from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings_obj: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings_obj or settings)

    async def run(self) -> str:
        # Provide file context if files are specified
        file_context = ""
        if self.task.files:
            file_contents = []
            for f in self.task.files:
                try:
                    content = await self.client.read_file(f)
                    file_contents.append(f"--- {f} ---\n{content}")
                except FileNotFoundError:
                    file_contents.append(f"--- {f} ---\n(File not found)")
            file_context = "\n\nFiles to work with:\n" + "\n".join(file_contents)

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{file_context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        # Try to parse structured response with file operations
        return await self._execute_response(response)

    async def _execute_response(self, response: str) -> str:
        """Parse response for file operations and execute them."""
        # Look for structured file operations in the response
        # Format: ```file:path\ncontent\n``` or similar
        try:
            # Try to find JSON block with operations
            if "```json" in response:
                json_start = response.index("```json") + 7
                json_end = response.index("```", json_start)
                json_str = response[json_start:json_end].strip()
                ops = json.loads(json_str)
                if isinstance(ops, dict) and "operations" in ops:
                    return await self._execute_operations(ops["operations"])
        except (ValueError, json.JSONDecodeError):
            pass

        # Fall back to returning the response as-is
        return response

    async def _execute_operations(self, operations: list[dict]) -> str:
        """Execute a list of file operations."""
        results = []
        for op in operations:
            op_type = op.get("type")
            path = op.get("path")
            content = op.get("content", "")

            if not path:
                results.append("Error: path is required")
                continue

            if op_type == "write":
                await self.client.write_file(path, content)
                results.append(f"Written: {path}")
            elif op_type == "read":
                file_content = await self.client.read_file(path)
                results.append(f"Read: {path} ({len(file_content)} chars)")
            elif op_type == "delete":
                import os
                os.remove(path)
                results.append(f"Deleted: {path}")
            elif op_type == "run":
                exit_code, stdout, stderr = await self.client.run_command(content.split())
                results.append(f"Command: {' '.join(content.split())} (exit={exit_code})")
                if stdout:
                    results.append(f"  stdout: {stdout[:500]}")
                if stderr:
                    results.append(f"  stderr: {stderr[:500]}")

        return "\n".join(results)
