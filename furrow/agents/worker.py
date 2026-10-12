from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 3

    async def run(self) -> str:
        logger.info("Starting worker", task_id=self.task.id, description=self.task.description[:100])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}"},
        ]

        for round_num in range(self.max_rounds):
            logger.debug("Worker round", round=round_num + 1, task_id=self.task.id)
            response = await self.client.complete(
                "\n".join(f"{m['role']}: {m['content']}" for m in messages),
                model=self.client.settings.worker_model,
            )

            # Try to parse tool calls from the response
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                # No tool calls, return the response as final result
                logger.info("Worker completed", task_id=self.task.id, rounds=round_num + 1)
                return response

            # Execute tool calls
            results = []
            for call in tool_calls:
                result = await self._execute_tool(call)
                results.append(result)

            # Add assistant response and tool results to messages
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "\n".join(results)})

        logger.warning("Max rounds reached", task_id=self.task.id, max_rounds=self.max_rounds)
        return "Max rounds reached. Last response: " + response

    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Parse tool calls from LLM response. Expected format: ```json\n{\"tool\": \"read_file\", \"args\": {\"path\": \"...\"}}\n```"""
        calls = []
        import re
        # Look for JSON code blocks
        pattern = r"```json\s*(\{.*?\})\s*```"
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                call = json.loads(match)
                if isinstance(call, dict) and "tool" in call:
                    calls.append(call)
            except json.JSONDecodeError:
                continue
        return calls

    async def _execute_tool(self, call: dict[str, Any]) -> str:
        tool = call.get("tool")
        args = call.get("args", {})

        try:
            if tool == "read_file":
                content = await self.client.read_file(args["path"])
                logger.debug("Read file", path=args["path"])
                return f"File content ({args['path']}):\n{content}"
            elif tool == "write_file":
                await self.client.write_file(args["path"], args["content"])
                logger.debug("Wrote file", path=args["path"])
                return f"Written to {args['path']}"
            elif tool == "list_files":
                files = self.client.list_files(args.get("directory", "."))
                logger.debug("Listed files", directory=args.get("directory", "."), count=len(files))
                return f"Files in {args.get('directory', '.')}:\n" + "\n".join(files)
            else:
                logger.warning("Unknown tool", tool=tool)
                return f"Unknown tool: {tool}"
        except Exception as e:
            logger.error("Tool execution failed", tool=tool, error=str(e))
            return f"Error executing {tool}: {e}"
