from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

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
        self.max_rounds = 5

    async def run(self) -> str:
        # Build initial prompt with task info
        files_info = f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}"
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n{files_info}\n"
        
        # Read any existing files mentioned in the task
        context_parts = []
        for file_path in self.task.files:
            try:
                content = await self.client.read_file(file_path)
                context_parts.append(f"--- {file_path} ---\n{content}")
            except FileNotFoundError:
                context_parts.append(f"--- {file_path} ---\n(File does not exist yet)")
        
        if context_parts:
            prompt += "\nExisting files:\n" + "\n\n".join(context_parts) + "\n"

        logger.info("worker_start", task_id=self.task.id, description=self.task.description[:100])
        
        # Tool-use loop
        for round_num in range(self.max_rounds):
            logger.debug("worker_round", task_id=self.task.id, round=round_num + 1)
            response = await self.client.complete(prompt, model=self.client.settings.worker_model)
            
            # Check for tool calls in the response
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                # No tool calls, return the response as summary
                logger.info("worker_complete", task_id=self.task.id, rounds=round_num + 1)
                return response.strip()
            
            # Execute tool calls
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("tool")
                args = tool_call.get("args", {})
                
                try:
                    if tool_name == "read_file":
                        result = await self.client.read_file(args["path"])
                        tool_results.append(f"read_file({args['path']}):\n{result}")
                        logger.debug("tool_read_file", path=args["path"])
                    elif tool_name == "write_file":
                        await self.client.write_file(args["path"], args["content"])
                        tool_results.append(f"write_file({args['path']}): OK")
                        logger.info("tool_write_file", path=args["path"], content_length=len(args["content"]))
                    elif tool_name == "list_files":
                        files = self.client.list_files(args["directory"])
                        tool_results.append(f"list_files({args['directory']}):\n" + "\n".join(files))
                        logger.debug("tool_list_files", directory=args["directory"], file_count=len(files))
                    else:
                        tool_results.append(f"Unknown tool: {tool_name}")
                        logger.warning("unknown_tool", tool=tool_name)
                except Exception as e:
                    tool_results.append(f"Error calling {tool_name}: {e}")
                    logger.error("tool_error", tool=tool_name, error=str(e))
            
            # Add tool results to prompt for next round
            prompt += f"\n\nAssistant: {response}\n\nTool results:\n" + "\n\n".join(tool_results) + "\n"

        logger.warning("worker_max_rounds_reached", task_id=self.task.id)
        return "Max rounds reached without completion."

    def _extract_tool_calls(self, response: str) -> list[dict]:
        """Extract tool calls from LLM response in JSON format."""
        tool_calls = []
        
        # Look for JSON tool calls in the format:
        # ```json
        # {"tool": "read_file", "args": {"path": "file.py"}}
        # ```
        json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        for block in json_blocks:
            try:
                data = json.loads(block)
                if "tool" in data:
                    tool_calls.append(data)
            except json.JSONDecodeError:
                continue
        
        # Also look for inline JSON objects with tool field
        # This handles cases where the LLM doesn't use code blocks
        inline_pattern = r'\{"tool"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*(\{.*?\})\}'
        for match in re.finditer(inline_pattern, response, re.DOTALL):
            try:
                tool_name = match.group(1)
                args = json.loads(match.group(2))
                tool_calls.append({"tool": tool_name, "args": args})
            except (json.JSONDecodeError, IndexError):
                continue
                
        return tool_calls
