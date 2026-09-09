from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT, WORKER_TOOL_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 5
        self.round = 0

    async def run(self) -> str:
        self.round = 0
        messages = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n\n{WORKER_TOOL_PROMPT}"},
        ]

        while self.round < self.max_rounds:
            self.round += 1
            response = await self.client.complete(
                json.dumps(messages),
                model=self.client.settings.worker_model,
            )
            
            try:
                data = json.loads(response)
                if "tool" in data:
                    result = await self._execute_tool(data["tool"], data.get("args", {}))
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": json.dumps({"tool_result": result})})
                    continue
                elif "done" in data:
                    return data.get("summary", "Task completed")
            except (json.JSONDecodeError, KeyError):
                # If not a tool call, treat as final response
                return response

        return "Max rounds reached without completion"

    async def _execute_tool(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "read_file":
            return await self.client.read_file(args["path"])
        elif tool == "write_file":
            await self.client.write_file(args["path"], args["content"])
            return f"File written: {args['path']}"
        elif tool == "list_files":
            return self.client.list_files(args.get("directory", "."))
        elif tool == "run_command":
            import asyncio
            cmd = args["command"]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            return {"stdout": stdout.decode(), "stderr": stderr.decode(), "returncode": proc.returncode}
        else:
            return f"Unknown tool: {tool}"
