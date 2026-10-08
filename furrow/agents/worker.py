from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_rounds: int = 3,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = max_rounds
    
    async def run(self) -> str:
        """Run the worker agent with tool-use loop."""
        system_prompt = WORKER_PROMPT
        
        # Build initial context with file contents if files are specified
        context = await self._build_initial_context()
        
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nTask: {self.task.description}\n\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nContext:\n{context}"}
        ]
        
        for round_num in range(self.max_rounds):
            response = await self.client.complete(
                "\n".join(m["content"] for m in messages),
                system=system_prompt,
                model=self.client.settings.worker_model,
            )
            
            # Check if the response contains tool calls
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # No tool calls, return the response as the result
                return response
            
            # Execute tool calls and add results to messages
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                messages.append({"role": "tool", "content": json.dumps({"result": result})})
            
            # Add assistant's response with tool calls
            messages.append({"role": "assistant", "content": response})
        
        return f"Completed {self.max_rounds} rounds. Last response: {response}"
    
    async def _build_initial_context(self) -> str:
        """Read initial file contents for context."""
        if not self.task.files:
            return "No specific files specified. Use list_files to explore."
        
        context_parts = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                context_parts.append(f"--- {file_path} ---\n{content}")
            except FileNotFoundError:
                context_parts.append(f"--- {file_path} ---\n[File does not exist]")
            except Exception as e:
                context_parts.append(f"--- {file_path} ---\n[Error reading: {e}]")
        
        return "\n\n".join(context_parts) if context_parts else "No files found."
    
    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Parse tool calls from LLM response. Expects JSON in a specific format."""
        tool_calls = []
        
        # Look for tool call patterns
        # Pattern: ```tool\n{"name": "tool_name", "arguments": {...}}\n```
        pattern = r"```tool\n(\{.*?\})\n```"
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        return tool_calls
    
    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a tool call and return the result."""
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})
        
        try:
            if name == "read_file":
                path = args.get("path", "")
                if not path:
                    return "Error: path is required"
                return await self.client.read_file(path)
            
            elif name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                if not path:
                    return "Error: path is required"
                await self.client.write_file(path, content)
                return f"Successfully wrote to {path}"
            
            elif name == "list_files":
                directory = args.get("directory", ".")
                files = self.client.list_files(directory)
                return json.dumps(files, indent=2)
            
            elif name == "run_command":
                # For running shell commands (tests, linting, etc.)
                import asyncio
                cmd = args.get("command", "")
                if not cmd:
                    return "Error: command is required"
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                return f"Exit code: {proc.returncode}\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
            
            else:
                return f"Error: Unknown tool '{name}'"
        
        except Exception as e:
            return f"Error executing {name}: {e}"