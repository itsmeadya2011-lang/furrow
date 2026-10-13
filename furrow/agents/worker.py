from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, WORKER_SYSTEM_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_iterations = 3

    async def run(self) -> str:
        # Read files to understand context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except FileNotFoundError:
                file_contents[file_path] = f"[File not found: {file_path}]"
            except Exception as e:
                file_contents[file_path] = f"[Error reading file: {e}]"

        # Build context
        context = ""
        if file_contents:
            context += "\nRelevant files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n{context}\n\nProceed with implementing the task. Use read_file, write_file, and list_files tools as needed."

        # Use a simple loop for tool calling (since we don't have native tool calling in the LLM client yet)
        # For now, we'll have the LLM output JSON with tool calls
        return await self._execute_with_tools(prompt)

    async def _execute_with_tools(self, prompt: str) -> str:
        """Execute the task with tool calling loop."""
        messages = [
            {"role": "system", "content": WORKER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        for iteration in range(self.max_iterations):
            response = await self.client.complete_with_messages(messages, model=self.client.settings.worker_model)

            # Parse tool calls from response
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                # No tool calls, return the response as summary
                return response

            # Execute tool calls
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "tool", "content": result, "name": tool_call["name"]})

        return "Max iterations reached without completion"

    def _parse_tool_calls(self, response: str) -> list[dict]:
        """Parse tool calls from LLM response. Expects JSON format."""
        tool_calls = []
        # Look for JSON tool calls in the response
        try:
            # Try to find JSON objects with tool calls
            import re
            # Pattern to match tool call JSON
            pattern = r'\{"tool":\s*"(\w+)",\s*"arguments":\s*(\{.*?\})\}'
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                tool_name = match[0]
                args = json.loads(match[1])
                tool_calls.append({"name": tool_name, "arguments": args})
        except Exception:
            pass
        return tool_calls

    async def _execute_tool(self, tool_call: dict) -> str:
        """Execute a tool call and return the result."""
        name = tool_call["name"]
        args = tool_call["arguments"]

        try:
            if name == "read_file":
                path = args.get("path", "")
                content = await self.client.read_file(path)
                return f"File content:\n{content}"
            elif name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                await self.client.write_file(path, content)
                return f"Successfully wrote to {path}"
            elif name == "list_files":
                directory = args.get("directory", ".")
                files = self.client.list_files(directory)
                return f"Files in {directory}:\n" + "\n".join(files)
            else:
                return f"Unknown tool: {name}"
        except Exception as e:
            return f"Error executing {name}: {e}"
