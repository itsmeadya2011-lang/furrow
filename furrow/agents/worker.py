from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_rounds: int = 3,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings
        self.max_rounds = max_rounds

    async def run(self) -> str:
        # Build initial prompt with file system context
        files_context = await self._get_files_context()

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Dependencies: {', '.join(self.task.dependencies) if self.task.dependencies else 'none'}\n\n"
            f"Current file system context:\n{files_context}\n\n"
            f"You have access to the following tools:\n"
            f"- read_file(path): Read a file's contents\n"
            f"- write_file(path, content): Write content to a file\n"
            f"- list_files(directory): List all files in a directory\n\n"
            f"Return your response as JSON with this shape:\n"
            f'{{\n'
            f'  "action": "tool" | "complete",\n'
            f'  "tool": "read_file" | "write_file" | "list_files",\n'
            f'  "args": {{...}},\n'
            f'  "summary": "Brief summary of what you did"\n'
            f'}}'
        )

        for round_num in range(self.max_rounds):
            try:
                response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            except Exception as e:
                logger.error("worker_llm_error", task_id=self.task.id, round=round_num, error=str(e))
                return f"LLM error after {round_num + 1} rounds: {e}"

            try:
                action = json.loads(response)
            except json.JSONDecodeError:
                # If not JSON, treat as completion
                logger.warning("worker_invalid_json", task_id=self.task.id, response=response[:200])
                return response

            if action.get("action") == "complete":
                summary = action.get("summary", "Task completed")
                logger.info("worker_completed", task_id=self.task.id, summary=summary)
                return summary

            if action.get("action") == "tool":
                tool_name = action.get("tool")
                args = action.get("args", {})

                try:
                    if tool_name == "read_file":
                        result = await self.client.read_file(args["path"])
                        prompt = f"File contents:\n{result}\n\nContinue with next action (JSON only):"
                    elif tool_name == "write_file":
                        await self.client.write_file(args["path"], args["content"])
                        prompt = f"File written successfully.\n\nContinue with next action (JSON only):"
                    elif tool_name == "list_files":
                        files = self.client.list_files(args["directory"])
                        prompt = f"Files in {args['directory']}:\n{', '.join(files)}\n\nContinue with next action (JSON only):"
                    else:
                        prompt = f"Unknown tool: {tool_name}\n\nContinue with next action (JSON only):"
                except FileNotFoundError:
                    prompt = f"File not found: {args.get('path', 'unknown')}\n\nContinue with next action (JSON only):"
                except PermissionError:
                    prompt = f"Permission denied: {args.get('path', 'unknown')}\n\nContinue with next action (JSON only):"
                except Exception as e:
                    logger.error("worker_tool_error", task_id=self.task.id, tool=tool_name, error=str(e))
                    prompt = f"Tool error: {e}\n\nContinue with next action (JSON only):"
            else:
                prompt = f"Invalid action. Return JSON only.\n\nContinue with next action (JSON only):"

        return "Max rounds reached. Task may be incomplete."

    async def _get_files_context(self) -> str:
        """Get context about relevant files for the task."""
        if not self.task.files:
            return "No specific files mentioned. Use list_files to explore."

        parts = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                parts.append(f"=== {file_path} ===\n{content[:2000]}")
            except FileNotFoundError:
                parts.append(f"=== {file_path} ===\n[File does not exist]")
            except Exception as e:
                logger.error("worker_read_context_error", task_id=self.task.id, file=file_path, error=str(e))
                parts.append(f"=== {file_path} ===\n[Error reading: {e}]")

        return "\n\n".join(parts) if parts else "No files found."