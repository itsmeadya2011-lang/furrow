from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


WORKER_SYSTEM_PROMPT = """You are a worker agent in an autonomous coding system called Furrow.

Your job is to implement the assigned task completely and concisely.

Available tools (output each tool call on its own line in this format):

```json
{
  "tool": "read_file",
  "args": {"path": "src/file.py"}
}
```

```json
{
  "tool": "write_file",
  "args": {"path": "src/file.py", "content": "new content"}
}
```

```json
{
  "tool": "list_files",
  "args": {"path": "src/"}
}
```

```json
{
  "tool": "run_command",
  "args": {"cmd": "pytest", "args": ["-q"]}
}
```

To call a tool, output exactly: TOOL: tool_name {"arg_key": "arg_value", ...}

Example:
TOOL: read_file {"path": "src/app.py"}
TOOL: list_files {"path": "."}

Rules:
- First read relevant files to understand the codebase before making changes.
- Then make targeted changes using write_file.
- Run tests to verify changes work.
- After running tests, return a summary of what was changed (files modified, commands run).
- When done, do NOT call any more tools — just return your summary as plain text.
- Work only on the assigned task. Do not refactor unrelated code.
- Make minimal, targeted changes.
- Do not spawn subagents.
"""

TOOL_PATTERN = re.compile(r"^TOOL:\s*(\w+)\s+(\{.*\})\s*$")

MAX_ROUNDS = 5


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self._changed_files: list[str] = []
        self._run_commands: list[str] = []

    async def run(self) -> str:
        files_info = ""
        if self.task.files:
            for f in self.task.files:
                try:
                    content = await self.client.read_file(f)
                    files_info += f"\n--- {f} ---\n{content}\n"
                except Exception:
                    files_info += f"\n--- {f} ---\n(file not found or unreadable)\n"

        conversation = (
            f"{WORKER_SYSTEM_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"\nExisting code:\n{files_info}\n"
        )

        for round_num in range(MAX_ROUNDS):
            response = await self.client.complete(
                conversation, system="", model=self.client.settings.worker_model
            )

            tool_matches = TOOL_PATTERN.findall(response, re.MULTILINE)
            if not tool_matches:
                return response.strip()

            for tool_name, args_str in tool_matches:
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    conversation += f"\n[Error: Could not parse arguments for {tool_name}]\n"
                    continue

                result = await self._execute_tool(tool_name, args)
                conversation += f"\n[Tool result: {tool_name}]\n{result}\n"

            if tool_matches:
                conversation += "\nYou have pending tool results. Provide your next action or final summary."

        return self._build_summary()

    async def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "read_file":
            path = args.get("path", "")
            try:
                content = await self.client.read_file(path)
                return content
            except Exception as e:
                return f"[Error reading {path}: {e}]"

        elif tool_name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            try:
                await self.client.write_file(path, content)
                if path not in self._changed_files:
                    self._changed_files.append(path)
                return f"[Wrote {path}]"
            except Exception as e:
                return f"[Error writing {path}: {e}]"

        elif tool_name == "list_files":
            directory = args.get("path", ".")
            try:
                files = self.client.list_files(directory)
                return "\n".join(files) if files else "(no files)"
            except Exception as e:
                return f"[Error listing {directory}: {e}]"

        elif tool_name == "run_command":
            cmd = args.get("cmd", "")
            cmd_args = args.get("args", [])
            cmd_str = f"{cmd} {' '.join(cmd_args)}".strip()
            try:
                result = await self.client.run_command(cmd, cmd_args)
                if cmd_str not in self._run_commands:
                    self._run_commands.append(cmd_str)
                return result
            except Exception as e:
                return f"[Error running {cmd_str}: {e}]"

        else:
            return f"[Unknown tool: {tool_name}]"

    def _build_summary(self) -> str:
        parts = []
        if self._changed_files:
            parts.append("Files modified:")
            for f in self._changed_files:
                parts.append(f"  - {f}")
        if self._run_commands:
            parts.append("Commands run:")
            for c in self._run_commands:
                parts.append(f"  - {c}")
        return "\n".join(parts) if parts else "No changes were made."

