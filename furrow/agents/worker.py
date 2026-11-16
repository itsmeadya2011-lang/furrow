from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from furrow.agents.command import run_command
from furrow.agents.prompts import WORKER_SYSTEM
from furrow.agents.security import (  # noqa: F401 - re-exported for backward compatibility
    _HOME_VAR_RE,
    _MAX_SUBSTITUTION_DEPTH,
    _REDIRECT_RE,
    _SEGMENT_SPLIT_RE,
    DENIED_COMMANDS,
    DENIED_PROGRAMS,
    MAX_LIST_ENTRIES,
    _expand_home,
    _extract_command_substitutions,
    _get_program_basename,
    _is_dangerous_segment,
    _is_force_push,
    _normalize_command,
    _redirects_outside_workspace,
    _shell_tokens,
    _split_command_segments,
    _target_inside_workspace,
    _text_is_denied,
    is_command_denied,
    resolve_in_workspace,
)
from furrow.agents.tools import FileTools, get_tool_definitions
from furrow.config import Settings, TaskModel, WorkerResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient, LLMResponse, ToolCall


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
        goal: str = "",
    ) -> None:
        self.task = task
        self.settings = settings if settings is not None else global_settings
        self.workspace = (
            Path(workspace) if workspace is not None else Path(self.settings.workspace)
        )
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.goal = goal
        self.files_changed: set[str] = set()
        self._tools = FileTools(
            client=self.client,
            settings=self.settings,
            workspace=self.workspace,
            files_changed=self.files_changed,
        )

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
                messages,
                tools=get_tool_definitions(),
                system=WORKER_SYSTEM,
                model=self.settings.worker_model,
            )
            if not response.tool_calls:
                summary = (
                    response.text.strip() if response.text.strip() else "Task completed."
                )
                return WorkerResult(
                    summary=summary,
                    success=True,
                    files_changed=sorted(self.files_changed),
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
                result = await self._execute_tool(tc)
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
            files_changed=sorted(self.files_changed),
            iterations=max_iters,
            error="worker iteration budget exhausted",
        )

    def _initial_text(self) -> str:
        parts = []
        if self.goal:
            parts.append(f"Overall goal: {self.goal}")
        parts.append(f"Task: {self.task.description}")
        parts.append(
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}"
        )
        parts.append(f"Workspace: {self.workspace}")
        return "\n".join(parts)

    async def _execute_tool(self, tc: ToolCall) -> str:
        name = tc.name
        args = self._coerce_tool_args(tc.arguments)
        if args is None:
            return "error: invalid tool arguments"
        if name == "read_file":
            return await self._tools.read_file(str(args.get("path", "")))
        if name == "write_file":
            return await self._tools.write_file(
                str(args.get("path", "")), str(args.get("content", ""))
            )
        if name == "list_files":
            path = args.get("path")
            return await self._tools.list_files(str(path) if path is not None else ".")
        if name == "run_command":
            return await run_command(
                str(args.get("command", "")), self.workspace, self.settings
            )
        return f"unknown tool: {name}"

    async def _read_file(self, path: str) -> str:
        return await self._tools.read_file(path)

    async def _write_file(self, path: str, content: str) -> str:
        return await self._tools.write_file(path, content)

    async def _list_files(self, path: str) -> str:
        return await self._tools.list_files(path)

    async def _run_command(self, command: str) -> str:
        return await run_command(command, self.workspace, self.settings)

    @staticmethod
    def _coerce_tool_args(arguments: Any) -> dict[str, Any] | None:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return None
            if isinstance(parsed, dict):
                return parsed
            return None
        return None