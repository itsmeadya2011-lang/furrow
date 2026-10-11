from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger, log_task_complete, log_task_failed, log_task_start

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.workspace = self.settings.workspace
        self.logger = get_logger("worker")

    async def run(self) -> str:
        """Execute the task by iteratively calling the LLM with tool use."""
        log_task_start(self.logger, self.task.id, self.task.description)
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "user", "content": self._build_initial_prompt()},
        ]

        max_rounds = 3
        for round_num in range(max_rounds):
            response = await self.client.complete(
                prompt=self._format_messages(messages),
                system=system_prompt,
                model=self.client.settings.worker_model,
            )

            # Parse tool calls from response
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                # No tool calls, return the response as result
                log_task_complete(self.logger, self.task.id, response)
                return response

            # Execute tool calls
            tool_results = []
            for call in tool_calls:
                result = await self._execute_tool(call)
                tool_results.append(result)

            # Add tool results to conversation
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": self._format_tool_results(tool_results)})

        result = "Max rounds reached. Last response: " + response[:500]
        log_task_failed(self.logger, self.task.id, result)
        return result

    def _build_system_prompt(self) -> str:
        return """You are a Furrow worker agent. Your job is to implement the assigned task by making actual code changes.

You have access to these tools:
- read_file(path): Read a file's contents
- write_file(path, content): Write content to a file
- list_files(directory): List files in a directory
- run_command(cmd, args...): Run a shell command

To use a tool, respond with a JSON object in this format:
{
  "tool": "tool_name",
  "args": {"arg1": "value1", "arg2": "value2"}
}

You can make multiple tool calls by responding with a JSON array of tool calls.
After each tool call, you'll receive the result and can continue.

Rules:
- Work only on the assigned task. Do not scope creep.
- Make minimal, targeted changes.
- Always read files before writing them.
- Run tests/lint after making changes if applicable.
- Return a concise summary when the task is complete.
- Do not spawn subagents."""

    def _build_initial_prompt(self) -> str:
        files_str = ", ".join(self.task.files) if self.task.files else "any relevant files"
        return f"""Task: {self.task.description}
Files to work on: {files_str}
Dependencies: {", ".join(self.task.dependencies) if self.task.dependencies else "none"}

Start by exploring the codebase to understand the structure, then implement the task.
When done, provide a summary of what you changed."""

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for the LLM (simple text format for non-tool-calling models)."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"{role.upper()}: {content}")
        return "\n\n".join(parts)

    def _parse_tool_calls(self, response: str) -> list[dict]:
        """Parse tool calls from LLM response."""
        # Try to find JSON in the response
        response = response.strip()

        # Check for code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "tool" in data:
                return [data]
        except json.JSONDecodeError:
            pass
        return []

    def _format_tool_results(self, results: list[dict]) -> str:
        parts = ["Tool results:"]
        for r in results:
            parts.append(f"- {r['tool']}: {r['result'][:500]}")
        return "\n".join(parts)

    async def _execute_tool(self, call: dict) -> dict:
        tool = call.get("tool")
        args = call.get("args", {})

        try:
            if tool == "read_file":
                path = self.workspace / args["path"]
                content = await self.client.read_file(path)
                return {"tool": tool, "result": content, "success": True}

            elif tool == "write_file":
                path = self.workspace / args["path"]
                content = args["content"]
                await self.client.write_file(path, content)
                return {
                    "tool": tool,
                    "result": f"Written {len(content)} bytes to {args['path']}",
                    "success": True,
                }

            elif tool == "list_files":
                directory = self.workspace / args.get("path", ".")
                files = self.client.list_files(directory)
                return {"tool": tool, "result": "\n".join(files), "success": True}

            elif tool == "run_command":
                cmd = args["cmd"]
                command_args = args.get("args", [])
                proc = await asyncio.create_subprocess_exec(
                    cmd,
                    *command_args,
                    cwd=self.workspace,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                result = stdout.decode() + stderr.decode()
                return {"tool": tool, "result": result, "success": proc.returncode == 0}

            else:
                return {"tool": tool, "result": f"Unknown tool: {tool}", "success": False}

        except (OSError, asyncio.TimeoutError, json.JSONDecodeError) as e:
            return {"tool": tool, "result": f"Error: {e!s}", "success": False}
