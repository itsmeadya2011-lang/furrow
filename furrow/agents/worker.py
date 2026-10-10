from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_LLM_RETRIES = 3


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_tool_rounds: int = 5,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_tool_rounds = max_tool_rounds

    async def run(self) -> str:
        # Build initial prompt with task description and workspace file listing
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        try:
            workspace_files = self.client.list_files(".")
            file_listing = "\n".join(workspace_files) if workspace_files else "(empty workspace)"
        except Exception:
            file_listing = "(unable to list workspace files)"

        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n\n"
            f"Current workspace files:\n{file_listing}\n"
        )

        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        final_answer = ""

        try:
            for _ in range(self.max_tool_rounds):
                response = await self._complete_with_retries(
                    prompt=messages[-1]["content"],
                    system=WORKER_PROMPT,
                )

                tool_call = self._parse_tool_call(response)
                if tool_call is None:
                    # Not a valid tool call — treat as final answer
                    final_answer = response
                    break

                tool_name = tool_call.get("tool")
                if tool_name == "done":
                    final_answer = tool_call.get("summary", "")
                    break

                # Execute the tool call
                try:
                    result = await self._execute_tool(tool_call)
                except Exception as e:
                    # Send the tool error back to the LLM so it can recover
                    result = f"Error executing tool: {e}"

                # Send the tool result back to the LLM
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool result:\n{result}"})
            else:
                # Reached max_tool_rounds without done
                final_answer = "Max tool rounds reached without completion."
        except Exception as e:
            # LLM call failed entirely after retries — return an error summary
            final_answer = (
                f"Worker error: LLM call failed after retries: "
                f"{type(e).__name__}: {e}"
            )

        return final_answer

    async def _complete_with_retries(self, prompt: str, system: str = "") -> str:
        """Call the LLM, retrying transient errors up to MAX_LLM_RETRIES times."""
        last_error: Exception | None = None
        for attempt in range(MAX_LLM_RETRIES + 1):
            try:
                return await self.client.complete(
                    prompt=prompt,
                    system=system,
                    model=self.client.settings.worker_model,
                )
            except Exception as e:
                last_error = e
                if attempt < MAX_LLM_RETRIES:
                    await asyncio.sleep(min(2 ** attempt, 10))
        raise RuntimeError(
            f"LLM call failed after {MAX_LLM_RETRIES + 1} attempts: {last_error}"
        )

    def _parse_tool_call(self, response: str) -> dict | None:
        """Try to extract a JSON tool call from the LLM response."""
        # Try to find a JSON object in the response
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict) or "tool" not in data:
            return None
        return data

    async def _execute_tool(self, tool_call: dict) -> str:
        """Execute a parsed tool call and return a string result."""
        tool_name = tool_call.get("tool")
        if tool_name == "read_file":
            path = tool_call.get("path", "")
            content = await self.client.read_file(path)
            return content
        elif tool_name == "write_file":
            path = tool_call.get("path", "")
            content = tool_call.get("content", "")
            await self.client.write_file(path, content)
            return f"Wrote {len(content)} bytes to {path}"
        elif tool_name == "list_files":
            directory = tool_call.get("directory", ".")
            files = self.client.list_files(directory)
            return "\n".join(files) if files else "(no files)"
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
