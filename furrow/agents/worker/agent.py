"""Worker agent for the Furrow autonomous coding system.

The worker agent implements assigned tasks by reading, writing, and running
code in the workspace using the available tools.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from furrow.agents.prompts import WORKER_SYSTEM, get_worker_tools
from furrow.agents.worker.tools import WorkerTools
from furrow.config import Settings, TaskModel, WorkerResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient, LLMResponse, ToolCall


@dataclass(slots=True)
class WorkerAgent:
    """Worker agent that executes tasks using tool calls."""

    task: TaskModel
    client: LLMClient | None = None
    settings: Settings | None = None
    workspace: Path | None = None
    goal: str = ""

    def __post_init__(self) -> None:
        self.settings = self.settings if self.settings is not None else global_settings
        self.workspace = Path(self.workspace) if self.workspace is not None else Path(self.settings.workspace)
        self.client = self.client if self.client is not None else LLMClient(settings=self.settings)
        self._tools = WorkerTools(self.client, self.settings, self.workspace)

    async def run(self) -> WorkerResult:
        try:
            return await self._run()
        except Exception as exc:  # pragma: no cover - defensive
            return WorkerResult(
                summary=str(exc), success=False, error=str(exc), iterations=0
            )

    async def _run(self) -> WorkerResult:
        max_iters = self.settings.max_worker_iterations
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": self._initial_text(),
            }
        ]
        iterations = 0
        for i in range(max_iters):
            iterations = i + 1
            response: LLMResponse = await self.client.chat(
                messages, tools=get_worker_tools(), system=WORKER_SYSTEM, model=self.settings.worker_model
            )
            if not response.tool_calls:
                summary = response.text.strip() if response.text.strip() else "Task completed."
                if response.stop_reason in TRUNCATED_STOP_REASONS:
                    return WorkerResult(
                        summary=summary,
                        success=False,
                        files_changed=sorted(self._tools.files_changed),
                        iterations=iterations,
                        error=f"worker response truncated (stop_reason={response.stop_reason})",
                    )
                return WorkerResult(
                    summary=summary,
                    success=True,
                    files_changed=sorted(self._tools.files_changed),
                    iterations=iterations,
                )
            # append assistant turn
            blocks: list[dict[str, Any]] = []
            if response.text:
                blocks.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            messages.append({"role": "assistant", "content": blocks})

            # execute tools
            result_blocks: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                result = await self._tools.execute(tc)
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                )
            messages.append({"role": "user", "content": result_blocks})

        # budget exhausted: the worker was truncated, so this is NOT a success.
        # Reporting success here would make the orchestrator mark the task as
        # completed instead of retrying it.
        return WorkerResult(
            summary="worker iteration budget exhausted",
            success=False,
            files_changed=sorted(self._tools.files_changed),
            iterations=max_iters,
            error="worker iteration budget exhausted",
        )

    def _initial_text(self) -> str:
        parts = []
        if self.goal:
            parts.append(f"Overall goal: {self.goal}")
        parts.append(f"Task: {self.task.description}")
        parts.append(f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}")
        parts.append(f"Workspace: {self.workspace}")
        return "\n".join(parts)


# Re-export for backward compatibility
from furrow.agents.worker.command_policy import (  # noqa: E402
    MAX_LIST_ENTRIES,
    MAX_READ_CHARS,
    TRUNCATED_STOP_REASONS,
    is_command_denied,
    resolve_in_workspace,
)