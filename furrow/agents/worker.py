from __future__ import annotations

import json
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
        # Get workspace context
        files_context = ""
        if self.task.files:
            for f in self.task.files:
                try:
                    content = await self.client.read_file(f)
                    files_context += f"\n--- {f} ---\n{content}\n"
                except Exception:
                    files_context += f"\n--- {f} ---\n(File not found or unreadable)\n"
        else:
            # List files in workspace
            files = self.client.list_files(self.client.settings.workspace)
            files_context = f"\nWorkspace files: {', '.join(files[:50])}"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{files_context}\n\nYou can use the following tools by responding with JSON:\n- {{'tool': 'read_file', 'path': 'path/to/file'}}\n- {{'tool': 'write_file', 'path': 'path/to/file', 'content': 'file content'}}\n- {{'tool': 'list_files', 'directory': 'path/to/dir'}}\n\nReturn a JSON object with 'action' and 'summary' when done, or tool calls to execute."
        
        max_rounds = 3
        for round_num in range(max_rounds):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # If not JSON, treat as final summary
                return response
            
            # Check if it's a tool call
            if "tool" in data:
                tool_result = await self._execute_tool(data)
                prompt = f"Tool result:\n{tool_result}\n\nContinue with next action or return final summary as JSON with 'action': 'done', 'summary': '...'"
            elif data.get("action") == "done":
                return data.get("summary", "Task completed")
            else:
                # Unexpected format, treat as summary
                return response
        
        return "Max rounds reached. Task may be incomplete."

    async def _execute_tool(self, tool_call: dict) -> str:
        tool = tool_call.get("tool")
        if tool == "read_file":
            path = tool_call.get("path")
            try:
                content = await self.client.read_file(path)
                return f"File content:\n{content}"
            except Exception as e:
                return f"Error reading file: {e}"
        elif tool == "write_file":
            path = tool_call.get("path")
            content = tool_call.get("content", "")
            try:
                await self.client.write_file(path, content)
                return f"File written: {path}"
            except Exception as e:
                return f"Error writing file: {e}"
        elif tool == "list_files":
            directory = tool_call.get("directory", ".")
            files = self.client.list_files(directory)
            return f"Files in {directory}: {', '.join(files)}"
        else:
            return f"Unknown tool: {tool}"
