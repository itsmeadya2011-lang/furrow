from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


WORKER_PROMPT = """You are a worker agent in an autonomous coding system called Furrow.

Your job is to implement the assigned task completely and concisely.

You have access to the following tools:
- read_file(path): Read a file's contents
- write_file(path, content): Write content to a file
- list_files(directory): List files in a directory

Rules:
- Work only on the assigned task. Do not refactor unrelated code.
- Make minimal, targeted changes.
- Return a concise summary of what you changed and any issues.
- Do not spawn subagents.

When you need to read or write files, output a JSON object with the tool call:
{
  "tool": "read_file",
  "args": {"path": "src/file.py"}
}

{
  "tool": "write_file",
  "args": {"path": "src/file.py", "content": "file contents"}
}

{
  "tool": "list_files",
  "args": {"directory": "src"}
}

After each tool call, you'll receive the result. Continue until the task is complete, then output a final summary without any tool calls."""


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 3

    async def run(self) -> str:
        # Read initial files for context
        file_contents = {}
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                file_contents[file_path] = content
            except Exception as e:  # noqa: BLE001
                logger.debug("file_read_failed", path=file_path, error=str(e))

        file_context = ""
        if file_contents:
            file_context = "\nCurrent file contents:\n"
            for path, content in file_contents.items():
                file_context += f"\n--- {path} ---\n{content}\n"

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}{file_context}\n"
        logger.info(
            "worker_start",
            task_id=self.task.id,
            description=self.task.description,
            files=self.task.files,
        )

        for round_num in range(self.max_rounds):
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)

            # Check if response contains a tool call
            tool_result = await self._execute_tool_call(response)
            if tool_result is not None:
                prompt += f"\nTool result: {tool_result}\n"
                continue

            # No tool call, this is the final response
            logger.info("worker_complete", task_id=self.task.id, rounds=round_num + 1)
            return response

        logger.warning("worker_max_rounds", task_id=self.task.id, max_rounds=self.max_rounds)
        return "Max rounds reached. Task may be incomplete."

    async def _execute_tool_call(self, response: str) -> str | None:
        try:
            # Try to parse as JSON tool call
            data = json.loads(response.strip())
            if not isinstance(data, dict) or "tool" not in data:
                return None

            tool = data["tool"]
            args = data.get("args", {})

            if tool == "read_file":
                path = args.get("path", "")
                try:
                    content = await self.client.read_file(path)
                    logger.debug("tool_read_file", path=path, size=len(content))
                    return f"File contents of {path}:\n{content}"
                except OSError as e:
                    logger.error("tool_read_file_failed", path=path, error=str(e))
                    return f"Error reading {path}: {e}"

            elif tool == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                try:
                    await self.client.write_file(path, content)
                    logger.info("tool_write_file", path=path, size=len(content))
                    return f"Successfully wrote to {path}"
                except OSError as e:
                    logger.error("tool_write_file_failed", path=path, error=str(e))
                    return f"Error writing {path}: {e}"

            elif tool == "list_files":
                directory = args.get("directory", ".")
                files = self.client.list_files(directory)
                logger.debug("tool_list_files", directory=directory, count=len(files))
                return f"Files in {directory}:\n" + "\n".join(files)

            logger.warning("tool_unknown", tool=tool)
            return f"Unknown tool: {tool}"
        except json.JSONDecodeError:
            # Not a tool call, return None to indicate final response
            return None
        except Exception as e:  # noqa: BLE001
            logger.error("tool_execution_failed", error=str(e))
            return f"Error executing tool: {e}"
