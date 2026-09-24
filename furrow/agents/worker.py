from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("furrow.worker")


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.max_rounds = 5

    async def run(self) -> str:
        logger.info("worker_started", task_id=self.task.id, description=self.task.description[:100])
        system_prompt = WORKER_PROMPT
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}",
            }
        ]

        for round_num in range(self.max_rounds):
            response = await self._chat(system_prompt, messages)

            # Check if response contains tool calls
            tool_calls = self._extract_tool_calls(response)
            if not tool_calls:
                # No tool calls, return the response as final result
                logger.info("worker_completed", task_id=self.task.id, rounds=round_num + 1)
                return response

            # Execute tool calls
            for tool_call in tool_calls:
                result = await self._execute_tool(tool_call)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "tool", "content": json.dumps({"result": result}), "tool_call_id": tool_call.get("id", "1")})

        logger.warning("worker_max_rounds_reached", task_id=self.task.id)
        return "Max rounds reached without completion"

    async def _chat(self, system: str, messages: list[dict[str, Any]]) -> str:
        # Use the LLM client's complete method with the full conversation
        conversation = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return await self.client.complete(conversation, system=system, model=self.settings.worker_model)

    def _extract_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Extract tool calls from the LLM response."""
        tool_calls = []
        # Look for tool call patterns in the response
        try:
            # Try to parse as JSON if it looks like a tool call
            data = json.loads(response)
            if isinstance(data, dict) and "tool" in data:
                tool_calls.append(data)
        except json.JSONDecodeError:
            # Try to find tool calls in the text
            pass
        return tool_calls

    async def _execute_tool(self, tool_call: dict[str, Any]) -> str:
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})

        logger.debug("tool_called", tool=tool_name, args=args)

        if tool_name == "read_file":
            path = args.get("path")
            if not path:
                return "Error: path is required for read_file"
            try:
                content = await self.client.read_file(path)
                logger.info("file_read", path=path, size=len(content))
                return content
            except Exception as e:
                logger.error("file_read_failed", path=path, error=str(e))
                return f"Error reading file: {e}"

        elif tool_name == "write_file":
            path = args.get("path")
            content = args.get("content", "")
            if not path:
                return "Error: path is required for write_file"
            try:
                await self.client.write_file(path, content)
                logger.info("file_written", path=path, size=len(content))
                return f"Successfully wrote to {path}"
            except Exception as e:
                logger.error("file_write_failed", path=path, error=str(e))
                return f"Error writing file: {e}"

        elif tool_name == "list_files":
            directory = args.get("directory", ".")
            try:
                files = self.client.list_files(directory)
                logger.info("files_listed", directory=directory, count=len(files))
                return "\n".join(files) if files else "No files found"
            except Exception as e:
                logger.error("file_list_failed", directory=directory, error=str(e))
                return f"Error listing files: {e}"

        logger.warning("unknown_tool", tool=tool_name)
        return f"Unknown tool: {tool_name}"
