from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


TOOL_PATTERN = re.compile(r'<tool>(\w+)</tool>\s*<args>(.*?)</args>', re.DOTALL)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 3

    async def run(self) -> str:
        # Initial prompt with task description
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        
        for round_num in range(self.max_rounds):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            
            # Check if the response contains tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls, return the response as summary
                return response.strip()
            
            # Execute tool calls and build results
            tool_results = []
            for tool_name, args_str in tool_calls:
                try:
                    args = json.loads(args_str)
                    result = await self._execute_tool(tool_name, args)
                    tool_results.append(f"<result>{tool_name}</result>\n<output>{result}</output>")
                except Exception as e:
                    tool_results.append(f"<result>{tool_name}</result>\n<output>Error: {e}</output>")
            
            # Feed results back to LLM for next round
            prompt = f"{response}\n\n" + "\n\n".join(tool_results) + "\n\nContinue working on the task."
        
        return "Max rounds reached. Task may be incomplete."

    def _parse_tool_calls(self, response: str) -> list[tuple[str, str]]:
        """Parse tool calls from LLM response."""
        matches = TOOL_PATTERN.findall(response)
        return [(name, args.strip()) for name, args in matches]

    async def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return the result."""
        if tool_name == "read_file":
            path = args.get("path", "")
            if not path:
                return "Error: path required"
            return await self.client.read_file(path)
        elif tool_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            if not path:
                return "Error: path required"
            await self.client.write_file(path, content)
            return f"Written to {path}"
        elif tool_name == "list_files":
            directory = args.get("directory", ".")
            files = self.client.list_files(directory)
            return "\n".join(files) if files else "No files found"
        else:
            return f"Error: Unknown tool {tool_name}"
