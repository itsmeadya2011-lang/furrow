from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import structlog

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)

__all__ = ["WorkerAgent"]


def _build_tools() -> list[dict[str, Any]]:
    """Return tool definitions compatible with both Anthropic and OpenAI formats."""
    read_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read"},
        },
        "required": ["path"],
    }
    write_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "File content to write"},
        },
        "required": ["path", "content"],
    }
    list_schema = {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "Directory to list"},
        },
        "required": ["directory"],
    }
    run_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run"},
        },
        "required": ["command"],
    }
    return [
        {
            "name": "read_file",
            "description": "Read the contents of a file",
            "input_schema": read_schema,
        },
        {
            "name": "write_file",
            "description": "Write or create a file with given content",
            "input_schema": write_schema,
        },
        {
            "name": "list_files",
            "description": "List all files in a directory recursively",
            "input_schema": list_schema,
        },
        {
            "name": "run_command",
            "description": "Run a shell command and return its output",
            "input_schema": run_schema,
        },
    ]


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_iterations = 10

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        tools = _build_tools()
        final_text = ""

        for _ in range(self.max_iterations):
            response = await self.client.complete_with_tools(
                prompt=messages[-1]["content"] if _ == 0 else "",
                tools=tools,
                system=WORKER_PROMPT,
                model=self.client.settings.worker_model,
            )

            # Anthropic response
            if hasattr(response, "content") and isinstance(response.content, list):
                tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
                text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
                if text_blocks:
                    final_text = text_blocks[0].text

                if not tool_uses:
                    break

                # Append assistant message
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool use and append results
                tool_results = []
                for tu in tool_uses:
                    result = await self._execute_tool(tu.name, tu.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tu.id,
                            "content": result,
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                # Update prompt for next iteration (use empty string; messages carry context)
                continue

            # OpenAI response
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)
            if tool_calls:
                messages.append(choice.message.model_dump())
                for tc in tool_calls:
                    result = await self._execute_tool(
                        tc.function.name, json.loads(tc.function.arguments)
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
                continue

            final_text = choice.message.content or ""
            break

        return final_text or "Task completed."

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "read_file":
                content = await self.client.read_file(args["path"])
                logger.info("tool_read_file", path=args["path"])
                return content
            elif name == "write_file":
                await self.client.write_file(args["path"], args["content"])
                logger.info("tool_write_file", path=args["path"])
                return f"Wrote {len(args['content'])} bytes to {args['path']}"
            elif name == "list_files":
                files = self.client.list_files(args["directory"])
                logger.info("tool_list_files", directory=args["directory"], count=len(files))
                return "\n".join(files) if files else "(empty directory)"
            elif name == "run_command":
                proc = await asyncio.create_subprocess_shell(
                    args["command"],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                except asyncio.TimeoutError:
                    proc.kill()
                    return "Command timed out after 120 seconds"
                out = (stdout or b"").decode() + (stderr or b"").decode()
                logger.info("tool_run_command", command=args["command"], exit_code=proc.returncode)
                return out or "(no output)"
            else:
                return f"Unknown tool: {name}"
        except Exception as e:  # noqa: BLE001
            logger.error("tool_error", tool=name, error=str(e))
            return f"Error executing {name}: {e}"