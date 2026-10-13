from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, Settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

console = Console()
logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)

    async def run(self) -> str:
        logger.info("Starting task", task_id=self.task.id, description=self.task.description)
        # Read relevant files first
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except Exception as e:
                file_contents[file_path] = f"[Error reading file: {e}]"

        # Build prompt with file context
        files_context = ""
        if file_contents:
            files_context = "\n\nRelevant files:\n"
            for path, content in file_contents.items():
                files_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{files_context}\n"
        
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        
        # Try to parse tool calls from response
        await self._execute_tool_calls(response)
        
        logger.info("Task completed", task_id=self.task.id)
        return response

    async def _execute_tool_calls(self, response: str) -> None:
        """Parse and execute tool calls from the LLM response."""
        # Look for JSON tool calls in the response
        # Format: ```json\n{"tool": "write_file", "path": "...", "content": "..."}\n```
        import re
        pattern = r'```json\n(\{.*?\})\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                tool_call = json.loads(match)
                await self._execute_tool(tool_call)
            except json.JSONDecodeError:
                continue

    async def _execute_tool(self, tool_call: dict) -> None:
        """Execute a single tool call."""
        tool = tool_call.get("tool")
        if tool == "write_file":
            path = tool_call.get("path")
            content = tool_call.get("content", "")
            if path:
                await self.client.write_file(path, content)
                console.print(f"[green]Written: {path}[/green]")
        elif tool == "read_file":
            path = tool_call.get("path")
            if path:
                content = await self.client.read_file(path)
                console.print(f"[blue]Read: {path}[/blue]")
        elif tool == "list_files":
            directory = tool_call.get("directory", ".")
            files = self.client.list_files(directory)
            console.print(f"[blue]Files in {directory}: {files}[/blue]")
