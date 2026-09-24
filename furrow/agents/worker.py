from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

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
        workspace: str | None = None,
        max_rounds: int = 3,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.max_rounds = max_rounds
        self.messages: list[dict[str, str]] = []

    async def run(self) -> str:
        self.messages = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": self._build_initial_prompt()},
        ]

        for round_num in range(self.max_rounds):
            response = await self.client.complete(
                self._format_messages(),
                model=self.client.settings.worker_model,
            )

            self.messages.append({"role": "assistant", "content": response})

            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                return self._extract_summary(response)

            tool_results = await self._execute_tool_calls(tool_calls)
            self.messages.append({"role": "user", "content": tool_results})

        return "Max rounds reached. Task may be incomplete."

    def _build_initial_prompt(self) -> str:
        files_list = self.client.list_files(self.workspace)
        files_str = "\n".join(f"  {f}" for f in files_list[:50]) if files_list else "  (empty)"

        return f"""Task: {self.task.description}
Files to touch: {', '.join(self.task.files) if self.task.files else 'any relevant files'}
Dependencies: {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}

Project structure (first 50 files):
{files_str}

You have access to these tools:
- read_file(path): Read a file's contents
- write_file(path, content): Write content to a file (creates directories as needed)
- edit_file(path, old_string, new_string): Replace a specific string in a file
- list_files(directory): List files in a directory
- run_command(cmd): Run a shell command (for tests, linting, etc.)

To use a tool, respond with a JSON block like:
```json
{{"tool": "read_file", "args": {{"path": "src/main.py"}}}}
```

You can make multiple tool calls in one response. After tool results are returned, you can continue working or provide a final summary.

Start by exploring the codebase to understand the task, then make your changes.
"""

    def _format_messages(self) -> str:
        return "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.messages)

    def _parse_tool_calls(self, response: str) -> list[dict]:
        pattern = r"```json\s*(\{.*?\})\s*```"
        matches = re.findall(pattern, response, re.DOTALL)
        tool_calls = []
        for match in matches:
            try:
                data = json.loads(match)
                if "tool" in data:
                    tool_calls.append(data)
            except json.JSONDecodeError:
                continue
        return tool_calls

    async def _execute_tool_calls(self, tool_calls: list[dict]) -> str:
        results = []
        for call in tool_calls:
            tool = call.get("tool")
            args = call.get("args", {})
            try:
                if tool == "read_file":
                    result = await self.client.read_file(self.workspace / args["path"])
                elif tool == "write_file":
                    await self.client.write_file(self.workspace / args["path"], args["content"])
                    result = f"Written to {args['path']}"
                elif tool == "edit_file":
                    result = await self._edit_file(args["path"], args["old_string"], args["new_string"])
                elif tool == "list_files":
                    files = self.client.list_files(self.workspace / args.get("directory", "."))
                    result = "\n".join(files) if files else "(empty)"
                elif tool == "run_command":
                    result = await self._run_command(args["cmd"])
                else:
                    result = f"Unknown tool: {tool}"
                results.append(f"Tool: {tool}\nResult:\n{result}")
            except Exception as e:
                results.append(f"Tool: {tool}\nError: {e}")
        return "\n\n---\n\n".join(results)

    async def _edit_file(self, path: str, old_string: str, new_string: str) -> str:
        full_path = self.workspace / path
        content = await self.client.read_file(full_path)
        if old_string not in content:
            return f"Error: old_string not found in {path}"
        new_content = content.replace(old_string, new_string, 1)
        await self.client.write_file(full_path, new_content)
        return f"Edited {path}"

    async def _run_command(self, cmd: list[str]) -> str:
        import asyncio

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.workspace,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            return f"Exit code: {proc.returncode}\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
        except asyncio.TimeoutError:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {e}"

    def _extract_summary(self, response: str) -> str:
        return response.strip()