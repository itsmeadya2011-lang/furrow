from __future__ import annotations

from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import Provider, TaskModel
from furrow.llm import LLMClient, CompletionResult, ToolCall

if TYPE_CHECKING:
    from furrow.config import Settings


# Universal tool definitions (provider-agnostic schema).
# `_get_tools()` converts these into the format expected by the current provider.
_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file at the given path, creating parent directories as needed.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write."},
                "content": {"type": "string", "description": "Content to write to the file."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files under the given directory (recursive).",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list files from.",
                },
            },
            "required": ["directory"],
        },
    },
]


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    def _get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions formatted for the current provider."""
        provider = self.client.settings.provider
        if provider == Provider.ANTHROPIC:
            return [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in _TOOL_DEFINITIONS
            ]
        # OpenAI (and other JSON-schema-based providers) use "parameters".
        return [dict(t) for t in _TOOL_DEFINITIONS]

    async def _execute_tool_call(self, tool_call: ToolCall) -> str:
        """Execute a single tool call and return its textual result."""
        if tool_call.name == "read_file":
            return await self.client.read_file(tool_call.arguments.get("path", ""))
        if tool_call.name == "write_file":
            await self.client.write_file(
                tool_call.arguments.get("path", ""),
                tool_call.arguments.get("content", ""),
            )
            return "ok"
        if tool_call.name == "list_files":
            return str(self.client.list_files(tool_call.arguments.get("directory", ".")))
        return f"Unknown tool: {tool_call.name}"

    async def run(self) -> str:
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\nFiles to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        tools = self._get_tools()
        result = await self.client.complete(prompt, model=self.client.settings.worker_model, tools=tools)

        if not isinstance(result, CompletionResult):
            return result

        if not result.tool_calls:
            return result.text or ""

        # Execute tool calls and collect results.
        tool_results: list[str] = []
        for tool_call in result.tool_calls:
            tool_results.append(
                f"<tool_result name=\"{tool_call.name}\">"
                f"{await self._execute_tool_call(tool_call)}"
                f"</tool_result>"
            )

        # Follow-up call so the LLM can produce a final summary based on tool results.
        followup_prompt = (
            f"{prompt}\n\n"
            f"Tool results:\n" + "\n".join(tool_results) + "\n\n"
            f"Based on the tool results above, provide a concise summary of what you changed."
        )
        followup = await self.client.complete(
            followup_prompt,
            model=self.client.settings.worker_model,
        )
        if isinstance(followup, CompletionResult):
            return followup.text or result.text or "Completed with tool calls"
        return followup or result.text or "Completed with tool calls"