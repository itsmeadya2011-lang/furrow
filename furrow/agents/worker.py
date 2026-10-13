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
        self.settings = settings
        self.max_iterations = 5
        self.iterations = 0

    async def run(self) -> str:
        self.iterations = 0
        
        # First, read any files mentioned in the task to understand context
        context = await self._gather_context()
        
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nContext:\n{context}\n"
        
        # Use a loop for iterative work
        while self.iterations < self.max_iterations:
            self.iterations += 1
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            
            # Try to parse as tool calls
            tool_calls = self._parse_tool_calls(response)
            if tool_calls:
                for call in tool_calls:
                    result = await self._execute_tool(call)
                    prompt += f"\n\nTool result: {result}"
                continue  # Let LLM decide next action
            
            # If no tool calls, assume work is done
            return response
        
        return "Max iterations reached. Task may need more cycles."

    async def _gather_context(self) -> str:
        """Read files mentioned in task to provide context."""
        context_parts = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                context_parts.append(f"=== {file_path} ===\n{content}")
            except Exception as e:
                context_parts.append(f"=== {file_path} ===\n[Could not read: {e}]")
        return "\n\n".join(context_parts) if context_parts else "No specific files provided."

    def _parse_tool_calls(self, response: str) -> list[dict]:
        """Parse tool calls from LLM response."""
        # Look for JSON tool calls in the response
        tool_calls = []
        lines = response.strip().split('\n')
        in_json = False
        json_buffer = []
        
        for line in lines:
            if line.strip().startswith('```json') or line.strip().startswith('```tool'):
                in_json = True
                json_buffer = []
                continue
            elif line.strip() == '```' and in_json:
                in_json = False
                try:
                    parsed = json.loads('\n'.join(json_buffer))
                    if isinstance(parsed, list):
                        tool_calls.extend(parsed)
                    elif isinstance(parsed, dict) and 'tool' in parsed:
                        tool_calls.append(parsed)
                except json.JSONDecodeError:
                    pass
                continue
            
            if in_json:
                json_buffer.append(line)
        
        # Also try to find inline JSON
        try:
            # Look for patterns like {"tool": "read_file", "args": {...}}
            import re
            matches = re.findall(r'\{[^}]*"tool"\s*:\s*"[^"]+"[^}]*\}', response)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if parsed not in tool_calls:
                        tool_calls.append(parsed)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
            
        return tool_calls

    async def _execute_tool(self, call: dict) -> str:
        """Execute a tool call and return result."""
        tool = call.get("tool")
        args = call.get("args", {})
        
        try:
            if tool == "read_file":
                path = args.get("path")
                if not path:
                    return "Error: read_file requires 'path' argument"
                return await self.client.read_file(path)
            
            elif tool == "write_file":
                path = args.get("path")
                content = args.get("content", "")
                if not path:
                    return "Error: write_file requires 'path' argument"
                await self.client.write_file(path, content)
                return f"Successfully wrote to {path}"
            
            elif tool == "list_files":
                directory = args.get("directory", ".")
                files = self.client.list_files(directory)
                return f"Files in {directory}:\n" + "\n".join(files)
            
            elif tool == "run_command":
                # For running shell commands
                import asyncio
                cmd = args.get("command")
                if not cmd:
                    return "Error: run_command requires 'command' argument"
                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                return f"stdout:\n{stdout.decode()}\nstderr:\n{stderr.decode()}"
            
            else:
                return f"Unknown tool: {tool}"
                
        except Exception as e:
            return f"Error executing {tool}: {e}"
