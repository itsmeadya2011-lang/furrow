from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


logger = logging.getLogger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.workspace = Path(self.settings.workspace).resolve()
        self._tool_results: list[str] = []

    async def run(self) -> str:
        # Build initial context with workspace info
        files_info = self._get_workspace_info()
        
        prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}
Workspace: {self.workspace}

Workspace files:
{files_info}

You have access to the following tools:
1. read_file(path) - Read a file's contents
2. write_file(path, content) - Write content to a file (creates directories)
3. edit_file(path, old_str, new_str) - Replace exact string in a file
4. list_files(directory) - List all files in a directory
5. run_command(cmd, args...) - Run a shell command in the workspace

Return a JSON object with your actions:
{{
  "actions": [
    {{"tool": "read_file", "path": "src/file.py"}},
    {{"tool": "write_file", "path": "src/file.py", "content": "..."}},
    {{"tool": "edit_file", "path": "src/file.py", "old_str": "...", "new_str": "..."}},
    {{"tool": "run_command", "cmd": ["pytest", "-q"]}}
  ],
  "summary": "What you did and any issues"
}}

You can make multiple tool calls in sequence. After each tool call, you'll receive the result and can continue.
"""
        
        # We'll use a simple loop for tool execution
        max_iterations = 10
        for iteration in range(max_iterations):
            response = await self.client.complete(
                prompt, 
                model=self.client.settings.worker_model,
                system="You are a coding agent that uses tools to complete tasks. Return only valid JSON."
            )
            
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # If not JSON, treat as final summary
                return response
            
            actions = data.get("actions", [])
            if not actions:
                # No more actions, return summary
                return data.get("summary", "Task completed")
            
            # Execute actions
            for action in actions:
                result = await self._execute_action(action)
                self._tool_results.append(f"Tool: {action.get('tool')}, Result: {result[:500]}")
            
            # Update prompt with results for next iteration
            prompt = f"""{WORKER_PROMPT}

Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}
Workspace: {self.workspace}

Previous tool results:
{chr(10).join(self._tool_results[-5:])}

Continue with more actions or return final summary as JSON with empty actions and a summary field."""
        
        return "Max iterations reached. " + "\n".join(self._tool_results)

    def _get_workspace_info(self) -> str:
        """Get a summary of the workspace structure."""
        try:
            files = self.client.list_files(self.workspace)
            # Limit to first 50 files
            return "\n".join(files[:50]) + ("\n..." if len(files) > 50 else "")
        except OSError as e:
            logger.warning("Failed to list workspace: %s", e)
            return "Unable to list workspace"

    async def _execute_action(self, action: dict[str, Any]) -> str:
        tool = action.get("tool")
        
        try:
            if tool == "read_file":
                path = self.workspace / action["path"]
                content = await self.client.read_file(path)
                return content
                
            elif tool == "write_file":
                path = self.workspace / action["path"]
                content = action["content"]
                await self.client.write_file(path, content)
                return f"Written {len(content)} bytes to {action['path']}"
                
            elif tool == "edit_file":
                path = self.workspace / action["path"]
                old_str = action["old_str"]
                new_str = action["new_str"]
                content = await self.client.read_file(path)
                if old_str not in content:
                    return f"ERROR: old_str not found in {action['path']}"
                new_content = content.replace(old_str, new_str, 1)
                await self.client.write_file(path, new_content)
                return f"Edited {action['path']}"
                
            elif tool == "list_files":
                directory = self.workspace / action.get("path", ".")
                files = self.client.list_files(directory)
                return "\n".join(files)
                
            elif tool == "run_command":
                cmd = action["cmd"]
                import asyncio
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=self.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                return f"Exit code: {proc.returncode}\nStdout: {stdout.decode()}\nStderr: {stderr.decode()}"
                
            else:
                return f"Unknown tool: {tool}"
                
        except asyncio.TimeoutError:
            return "ERROR: Command timed out"
        except OSError as e:
            logger.error("Tool execution failed: %s", e)
            return f"ERROR: {e!s}"