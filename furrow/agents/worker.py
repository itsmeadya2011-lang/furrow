from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

# Maximum tool-use rounds per task. The architecture targets 1-3 rounds; this cap
# prevents runaway loops while giving the worker enough iterations to finish.
TOOL_MAX_ROUNDS = 10


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files = ", ".join(self.task.files) if self.task.files else "any"

        system = WORKER_PROMPT
        conversation = (
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n\n"
            "Available tools (respond with exactly one JSON tool-call object per turn):\n"
            "- read_file: read a file's contents. args: {\"path\": \"src/foo.py\"}\n"
            '- write_file: create or overwrite a file. args: {\"path\": "src/foo.py", "content": "..."}\n'
            '- list_files: list files under a directory. args: {\"path\": "src"}\n'
            '- done: signal completion. args: {\"summary": "brief summary of changes"}\n\n'
            "Continue calling tools until you issue a 'done' tool call.\n"
        )

        for round_num in range(1, TOOL_MAX_ROUNDS + 1):
            response = await self.client.complete(
                conversation, system=system, model=self.client.settings.worker_model
            )
            calls = self.client.extract_tool_calls(response)
            if not calls:
                conversation += (
                    f"\nTurn {round_num}:\n> (no valid tool call)\n"
                    "< Please respond with a single JSON tool-call object.\n"
                )
                continue

            call = calls[0]
            tool = call.get("tool")
            args = call.get("args", {}) or {}

            if tool == "done":
                return args.get("summary", "Task completed.")

            result = await self._execute_tool(tool, args)
            conversation += (
                f"\nTurn {round_num}:\n> {json.dumps(call)}\n< {result}\n"
            )

        return f"Worker reached maximum tool rounds ({TOOL_MAX_ROUNDS}) without completion."

    async def _execute_tool(self, tool: str, args: dict[str, Any]) -> str:
        try:
            if tool == "read_file":
                return await self.client.read_file(args["path"])
            if tool == "write_file":
                await self.client.write_file(args["path"], args["content"])
                return f"Wrote {args['path']}"
            if tool == "list_files":
                files = self.client.list_files(args.get("path", "."))
                return "\n".join(files) if files else "(empty)"
            return f"Unknown tool: {tool}"
        except KeyError as e:
            return f"Error: missing required argument {e}"
        except Exception as e:  # noqa: BLE001 - surface tool errors to the worker
            return f"Error: {e}"
