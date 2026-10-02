from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import CompletionResult, LLMClient, ToolCall

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    """Executes a single task using LLM tool calling in an agentic loop.

    The worker sends a task prompt to the LLM with file-editing tools enabled.
    When the LLM returns tool calls, the worker executes them and feeds the
    results back as conversation context in subsequent rounds.  This repeats
    until the LLM returns a final text response (no tool calls) or the
    maximum number of rounds is reached.
    """

    AVAILABLE_TOOLS: ClassVar[list[str]] = ["read_file", "write_file", "list_files", "edit_file"]
    MAX_ROUNDS = 10

    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files_str = (
            ", ".join(self.task.files) if self.task.files else "any files in the workspace"
        )
        prompt = (
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_str}\n"
        )

        for _round in range(1, self.MAX_ROUNDS + 1):
            result = await self.client.complete_with_tools(
                prompt=prompt,
                system=WORKER_PROMPT,
                model=self.client.settings.worker_model,
                tools=self.AVAILABLE_TOOLS,
            )

            # No tool calls → final response
            if not result.tool_calls:
                return result.text or f"Task '{self.task.description}' is complete."

            # Execute all tool calls in this round
            tool_outputs = await self._execute_tool_calls(result.tool_calls)

            # Append tool results to the conversation prompt
            prompt += self._format_tool_results(result, tool_outputs)

        return (
            f"Reached max rounds ({self.MAX_ROUNDS}) for task: "
            f"{self.task.description}. The task may be incomplete."
        )

    async def _execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[str]:
        """Execute tool calls and return their string results."""
        results: list[str] = []
        for call in tool_calls:
            try:
                output = await self.client.execute_tool(call)
            except Exception as exc:  # noqa: BLE001
                output = f"Error executing {call.name}: {exc}"
            results.append(output)
        return results

    @staticmethod
    def _format_tool_results(
        result: CompletionResult, tool_outputs: list[str]
    ) -> str:
        """Format tool calls and their results as conversation context."""
        parts = ["\n\n--- Tool execution results ---\n"]
        for call, output in zip(result.tool_calls, tool_outputs):
            params = json.dumps(call.input)
            parts.append(
                f"\n[executed] {call.name}({params})\n"
                f"Result: {output}"
            )
        parts.append(
            "\n\nThe above tool calls have already been executed. "
            "Do not repeat them. Continue working on the task — make "
            "additional tool calls if needed, or return a concise summary "
            "when the task is complete."
        )
        return "".join(parts)
