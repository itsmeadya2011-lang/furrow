from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT, WORKER_TOOLS_PROMPT
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
        # First, read any relevant files to understand context
        context = await self._gather_context()

        prompt = f"{WORKER_PROMPT}\n\n{WORKER_TOOLS_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\nCurrent context:\n{context}\n\nYou have up to {self.max_rounds} rounds to complete this task. Respond with either:\n1. A tool call in JSON format: {{\"tool\": \"read|write|list|bash\", \"args\": {{...}}}}\n2. A final result: {{\"done\": true, \"summary\": \"...\"}}"

        for round_num in range(self.max_rounds):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            try:
                action = json.loads(response)
            except json.JSONDecodeError:
                # If not JSON, treat as final summary
                return response.strip()

            if action.get("done"):
                return action.get("summary", "Task completed")

            # Execute tool
            result = await self._execute_tool(action)
            prompt = f"Tool result:\n{result}\n\nContinue with next action or return {{\"done\": true, \"summary\": \"...\"}}"

        return "Max rounds reached. Task may be incomplete."

    async def _gather_context(self) -> str:
        if not self.task.files:
            return "No specific files specified."
        context_parts = []
        for f in self.task.files:
            try:
                content = await self.client.read_file(f)
                context_parts.append(f"=== {f} ===\n{content}")
            except Exception:
                context_parts.append(f"=== {f} === (not found)")
        return "\n\n".join(context_parts)

    async def _execute_tool(self, action: dict) -> str:
        tool = action.get("tool")
        args = action.get("args", {})

        try:
            if tool == "read":
                path = args.get("path")
                if not path:
                    return "Error: read tool requires 'path' argument"
                content = await self.client.read_file(path)
                return f"File content:\n{content}"

            elif tool == "write":
                path = args.get("path")
                content = args.get("content")
                if not path or content is None:
                    return "Error: write tool requires 'path' and 'content' arguments"
                await self.client.write_file(path, content)
                return f"Successfully wrote to {path}"

            elif tool == "list":
                path = args.get("path", ".")
                files = self.client.list_files(path)
                return f"Files in {path}:\n" + "\n".join(files)

            elif tool == "bash":
                import asyncio
                cmd = args.get("command")
                if not cmd:
                    return "Error: bash tool requires 'command' argument"
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                return f"Exit code: {proc.returncode}\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"

            else:
                return f"Error: Unknown tool '{tool}'"

        except asyncio.TimeoutError:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error executing tool: {e}"
