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
        self.max_rounds = 3

    async def run(self) -> str:
        # Build initial prompt with available tools
        prompt = self._build_prompt()
        
        for round_num in range(self.max_rounds):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            
            # Try to parse as tool calls
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                # No tool calls - assume task is complete, return summary
                return response
            
            # Execute tool calls
            results = []
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                results.append(result)
            
            # Build next prompt with tool results
            prompt = self._build_prompt(tool_results=results)
        
        return "Max rounds reached. Task may be incomplete."

    def _build_prompt(self, tool_results: list[str] | None = None) -> str:
        base = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Available tools:
- read_file(path): Read a file's contents
- write_file(path, content): Write content to a file
- list_files(directory): List files in a directory

Respond with either:
1. A summary of what you did (when task is complete)
2. JSON tool calls in this format:
[
  {{"tool": "read_file", "args": {{"path": "src/file.py"}}}},
  {{"tool": "write_file", "args": {{"path": "src/file.py", "content": "print('hello')"}}}}
]"""

        if tool_results:
            base += "\n\nTool results:\n" + "\n".join(f"- {r}" for r in tool_results)
        
        return base

    def _parse_tool_calls(self, response: str) -> list[dict] | None:
        response = response.strip()
        if response.startswith('['):
            try:
                calls = json.loads(response)
                if isinstance(calls, list) and all(isinstance(c, dict) and 'tool' in c for c in calls):
                    return calls
            except json.JSONDecodeError:
                pass
        return None

    async def _execute_tool(self, tool_call: dict) -> str:
        tool = tool_call.get("tool")
        args = tool_call.get("args", {})
        
        try:
            if tool == "read_file":
                content = await self.client.read_file(args["path"])
                return f"read_file({args['path']}): {content[:500]}{'...' if len(content) > 500 else ''}"
            elif tool == "write_file":
                await self.client.write_file(args["path"], args["content"])
                return f"write_file({args['path']}): OK ({len(args['content'])} chars)"
            elif tool == "list_files":
                files = self.client.list_files(args["path"])
                return f"list_files({args['path']}): {', '.join(files)}"
            else:
                return f"Unknown tool: {tool}"
        except Exception as e:
            return f"Error executing {tool}: {e}"
