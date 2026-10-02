from __future__ import annotations

import json
import re
from typing import Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

# Maximum tool-call rounds per worker execution
_DEFAULT_MAX_ITERS = 10


class WorkerAgent:
    """Agent that implements a single task using an LLM tool-call loop.

    The worker receives a task description and a list of files to touch,
    then iteratively:
      1. Sends its current context + task to the LLM.
      2. Parses the response for tool-use instructions (JSON blocks).
      3. Executes file operations (read, write, list, edit) locally.
      4. Feeds results back to the LLM.
      5. Stops when the LLM signals completion or max iterations is hit.

    Supported tool-call format (JSON in a ```tool block or plain JSON):
        {"tool": "read_file", "path": "src/module.py"}
        {"tool": "write_file", "path": "src/module.py", "content": "..."}
        {"tool": "edit_file", "path": "src/module.py", "old": "...", "new": "..."}
        {"tool": "list_files", "path": "src/"}
        {"tool": "done", "summary": "Finished implementing X"}
    """

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_iterations: int = _DEFAULT_MAX_ITERS,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.max_iterations = max_iterations

    async def run(self) -> str:
        """Execute the task and return a summary of changes."""
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": WORKER_PROMPT
                + """

You have access to the following tools. To use a tool, output a JSON object
with a "tool" key. Available tools:
- read_file: {"tool": "read_file", "path": "<path>"}
- write_file: {"tool": "write_file", "path": "<path>", "content": "<content>"}
- edit_file: {"tool": "edit_file", "path": "<path>", "old": "<old text>", "new": "<new text>"}
- list_files: {"tool": "list_files", "path": "<path or dir>"}
- done: {"tool": "done", "summary": "<what you accomplished>"}

After each tool result, continue working. When the task is complete, call "done".
You may call multiple tools in one response by emitting multiple JSON blocks.
""",
            },
            {
                "role": "user",
                "content": (
                    f"Task: {self.task.description}\n"
                    f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any relevant files'}\n"
                    f"Workspace: {self.client.settings.workspace}\n"
                    "Start by exploring the workspace to understand the current state."
                ),
            },
        ]

        summary = ""
        for iteration in range(self.max_iterations):
            response = await self.client.converse(messages)

            # Extract tool calls from response
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                # No tools requested — treat as final answer
                summary = response.strip()
                break

            all_done = False
            for tc in tool_calls:
                tool_name = tc.get("tool", "")

                if tool_name == "done":
                    summary = tc.get("summary", response.strip())
                    all_done = True
                else:
                    result = await self._execute_tool(tool_name, tc)
                    messages.append({"role": "assistant", "content": json.dumps(tc)})
                    messages.append({"role": "user", "content": result})

            if all_done:
                break

            # If the model didn't emit any tool calls at all, it's done
            if not any(tc.get("tool") != "done" for tc in tool_calls):
                summary = response.strip()
                break

            # Limit message history to avoid excessive token usage
            if len(messages) > 40:
                messages = messages[:2] + messages[-20:]

        return summary or "Worker completed without a summary."

    # ------------------------------------------------------------------
    # Tool call parsing
    # ------------------------------------------------------------------

    def _extract_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Extract tool-call JSON objects from an LLM response.

        Supports both code-block-wrapped JSON and inline JSON.
        """
        calls: list[dict[str, Any]] = []

        # Try to find JSON blocks (within ```tool, ```json, or plain ```)
        # Pattern: optional language tag, then JSON object(s)
        block_pattern = r"```(?:tool|json|)?\n(\{.*?\})```"
        blocks = re.findall(block_pattern, response, re.DOTALL)

        if not blocks:
            # Try plain JSON objects wrapped in backticks without language tag
            plain_blocks = re.findall(r"```\n(\{.*?\})```", response, re.DOTALL)
            blocks = plain_blocks

        if not blocks:
            # Try raw JSON objects (not in code blocks)
            # Look for objects with "tool" key
            raw_pattern = r'\{[^{}]*"tool"\s*:\s*"[^"]+"[^{}]*\}'
            raw_matches = re.findall(raw_pattern, response, re.DOTALL)
            blocks = raw_matches

        for block in blocks:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict) and "tool" in parsed:
                    calls.append(parsed)
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "tool" in item:
                            calls.append(item)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    cleaned = (
                        block.replace("None", "null")
                        .replace("True", "true")
                        .replace("False", "false")
                    )
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, dict) and "tool" in parsed:
                        calls.append(parsed)
                except json.JSONDecodeError:
                    continue

        return calls

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a single tool and return the result string."""
        try:
            if tool_name == "read_file":
                path = args.get("path", "")
                if not path:
                    return "[tool error: read_file requires 'path']"
                content = await self.client.read_file_safe(path)
                if content is None:
                    return f"[file does not exist: {path}]"
                # Truncate very large files
                if len(content) > 8000:
                    content = content[:8000] + f"\n... (truncated, {len(content)} total chars)"
                return f"[file: {path}]\n{content}"

            elif tool_name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                if not path:
                    return "[tool error: write_file requires 'path' and 'content']"
                await self.client.write_file(path, content)
                return f"[wrote file: {path} ({len(content)} chars)]"

            elif tool_name == "edit_file":
                path = args.get("path", "")
                old = args.get("old", "")
                new = args.get("new", "")
                if not path or not old:
                    return "[tool error: edit_file requires 'path', 'old', and 'new']"
                existing = await self.client.read_file_safe(path)
                if existing is None:
                    return f"[file does not exist: {path}]"
                if old not in existing:
                    return f"[old text not found in {path}]"
                new_content = existing.replace(old, new, 1)
                await self.client.write_file(path, new_content)
                return f"[edited file: {path}]"

            elif tool_name == "list_files":
                path = args.get("path", ".")
                files = await self.client.list_files(path)
                if not files:
                    return f"[no files in {path}]"
                return f"[files in {path}]\n" + "\n".join(files)

            else:
                return f"[unknown tool: {tool_name}]"

        except (OSError, ValueError, TypeError) as e:
            return f"[tool error: {e}]"
