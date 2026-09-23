from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_rounds = 5

    async def run(self) -> str:
        logger.info("worker.started", task_id=self.task.id, description=self.task.description[:50])
        messages = [
            {"role": "system", "content": WORKER_PROMPT},
            {"role": "user", "content": f"Task: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}"}
        ]

        for round_num in range(self.max_rounds):
            logger.debug("worker.round", task_id=self.task.id, round=round_num + 1)
            response = await self.client.complete_with_tools(
                messages=messages,
                model=self.client.settings.worker_model,
                tools=self._get_tools()
            )

            # Check if the response contains tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                messages.append({"role": "assistant", "content": response.content or "", "tool_calls": response.tool_calls})
                
                for tool_call in response.tool_calls:
                    logger.debug("worker.tool_call", task_id=self.task.id, tool=tool_call.function.name)
                    result = await self._execute_tool(tool_call)
                    messages.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
            else:
                # No tool calls, task is done
                logger.info("worker.completed", task_id=self.task.id, rounds=round_num + 1)
                return response.content or "Task completed"

        logger.warning("worker.max_rounds_reached", task_id=self.task.id)
        return "Max rounds reached. Task may be incomplete."

    def _get_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to read"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to the file to write"},
                            "content": {"type": "string", "description": "Content to write to the file"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "description": "Directory to list files from"}
                        },
                        "required": ["directory"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to run"},
                            "cwd": {"type": "string", "description": "Working directory (optional)"}
                        },
                        "required": ["command"]
                    }
                }
            }
        ]

    async def _execute_tool(self, tool_call: Any) -> str:
        import asyncio
        
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        try:
            if name == "read_file":
                return await self.client.read_file(args["path"])
            elif name == "write_file":
                await self.client.write_file(args["path"], args["content"])
                return f"Successfully wrote to {args['path']}"
            elif name == "list_files":
                files = self.client.list_files(args["directory"])
                return "\n".join(files) if files else "No files found"
            elif name == "run_command":
                proc = await asyncio.create_subprocess_shell(
                    args["command"],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=args.get("cwd")
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                return f"Exit code: {proc.returncode}\nStdout:\n{stdout.decode()}\nStderr:\n{stderr.decode()}"
            else:
                return f"Unknown tool: {name}"
        except asyncio.TimeoutError:
            logger.warning("worker.tool_timeout", tool=name)
            return f"Tool {name} timed out"
        except Exception as e:
            logger.error("worker.tool_error", tool=name, error=str(e))
            return f"Tool {name} error: {str(e)}"