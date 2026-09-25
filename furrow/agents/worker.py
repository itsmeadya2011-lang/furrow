from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()
        self.workspace = self.settings.workspace
        self.max_rounds = 3

    async def run(self) -> str:
        """Run the worker agent to complete the assigned task."""
        system_prompt = self._build_system_prompt()

        messages = [
            {
                "role": "user",
                "content": self._build_initial_prompt(),
            }
        ]

        for round_num in range(self.max_rounds):
            response = await self.client.complete(
                prompt="",  # We pass messages directly via the API
                system=system_prompt,
                model=self.client.settings.worker_model,
            )

            # Parse tool calls from response
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                # No tool calls, task might be complete
                return response

            # Execute tool calls
            tool_results = []
            for call in tool_calls:
                result = await self._execute_tool(call)
                tool_results.append(result)

            # Add tool results to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "\n".join(tool_results)})

        return "Max rounds reached. Task may be incomplete."

    def _build_system_prompt(self) -> str:
        return f"""{WORKER_PROMPT}

You have access to the following tools:

1. read_file(path) - Read a file's contents
2. write_file(path, content) - Write content to a file (creates directories)
3. edit_file(path, old_str, new_str) - Replace a string in a file
4. list_files(directory) - List files in a directory
5. run_command(cmd) - Run a shell command in the workspace

When you need to use a tool, output a JSON object like:
{{
  "tool": "read_file",
  "args": {{"path": "src/main.py"}}
}}

You can make multiple tool calls in one response by outputting multiple JSON objects, each on its own line.

Your workspace is: {self.workspace}
Your task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}

Rules:
- Work only on the assigned task. Do not scope creep.
- Make minimal, targeted changes.
- Run tests/lint if your changes affect existing code.
- Return a concise summary when done.
"""

    def _build_initial_prompt(self) -> str:
        files = self.client.list_files(self.workspace)
        context = f"Workspace files:\n" + "\n".join(f"  {f}" for f in files[:30]) if files else "Empty workspace."

        return f"""Task: {self.task.description}

Files to modify: {', '.join(self.task.files) if self.task.files else 'Determine appropriate files'}

{context}

Start by exploring the relevant files, then implement the task. When complete, provide a summary of what you changed."""

    def _parse_tool_calls(self, text: str) -> list[dict]:
        """Parse tool calls from LLM response."""
        calls = []
        # Find JSON objects in the text
        pattern = r'\{[^{}]*"tool"\s*:\s*"[^"]+"[^{}]*\}'
        for match in re.finditer(pattern, text):
            try:
                call = json.loads(match.group())
                if "tool" in call and "args" in call:
                    calls.append(call)
            except json.JSONDecodeError:
                continue
        return calls

    async def _execute_tool(self, call: dict) -> str:
        tool = call.get("tool")
        args = call.get("args", {})

        try:
            if tool == "read_file":
                path = self.workspace / args["path"]
                content = await self.client.read_file(path)
                return f"File content ({args['path']}):\n{content}"

            elif tool == "write_file":
                path = self.workspace / args["path"]
                await self.client.write_file(path, args["content"])
                return f"Written to {args['path']}"

            elif tool == "edit_file":
                path = self.workspace / args["path"]
                content = await self.client.read_file(path)
                old_str = args["old_str"]
                new_str = args["new_str"]
                if old_str not in content:
                    return f"Error: Could not find '{old_str}' in {args['path']}"
                new_content = content.replace(old_str, new_str, 1)
                await self.client.write_file(path, new_content)
                return f"Edited {args['path']}"

            elif tool == "list_files":
                directory = args.get("directory", ".")
                path = self.workspace / directory
                files = self.client.list_files(path)
                return f"Files in {directory}:\n" + "\n".join(f"  {f}" for f in files)

            elif tool == "run_command":
                import asyncio
                cmd = args["cmd"]
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    cwd=self.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                result = stdout.decode() + stderr.decode()
                return f"Command: {cmd}\nExit code: {proc.returncode}\nOutput:\n{result}"

            else:
                return f"Error: Unknown tool '{tool}'"

        except asyncio.TimeoutError:
            return f"Error: Command timed out"
        except Exception as e:
            return f"Error executing {tool}: {e}"