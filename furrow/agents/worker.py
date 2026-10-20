from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT, WORKER_TOOL_INSTRUCTIONS
from furrow.config import TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient, ToolCall

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_TOOL_ROUNDS = 8

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a file relative to the workspace root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a file relative to the workspace root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List files under an optional directory.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        },
    },
]


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings or default_settings)
        self.workspace = workspace if workspace is not None else self.client.settings.workspace

    def _safe_path(self, raw: str) -> Path | None:
        if raw is None:
            return None
        candidate = (self.workspace / raw).resolve()
        try:
            candidate.relative_to(self.workspace.resolve())
        except ValueError:
            return None
        if Path(raw).is_absolute():
            # allow only if inside workspace
            try:
                Path(raw).resolve().relative_to(self.workspace.resolve())
            except ValueError:
                return None
        return candidate

    async def _execute_tool(self, call: ToolCall) -> str:
        if call.name == "read_file":
            path = self._safe_path(call.arguments.get("path", ""))
            if path is None:
                return "Refused: path escapes workspace."
            if not path.exists():
                return f"File not found: {path}"
            try:
                text = await self.client.read_file(path)
            except Exception as e:  # noqa: BLE001
                return f"Error reading file: {e}"
            if len(text) > 20000:
                text = text[:20000] + "\n...[truncated]"
            return text
        if call.name == "write_file":
            path = self._safe_path(call.arguments.get("path", ""))
            if path is None:
                return "Refused: path escapes workspace."
            content = call.arguments.get("content", "")
            path.parent.mkdir(parents=True, exist_ok=True)
            await self.client.write_file(path, content)
            return f"Wrote {path.relative_to(self.workspace.resolve())}"
        if call.name == "list_files":
            raw = call.arguments.get("path", "") or ""
            base = self.workspace if raw == "" else self._safe_path(raw)
            if base is None:
                return "Refused: path escapes workspace."
            if not base.exists():
                return "Directory not found."
            files = self.client.list_files(base)
            return "\n".join(files)
        return f"Unknown tool: {call.name}"

    async def run(self) -> str:
        prompt = (
            f"{WORKER_PROMPT}\n{WORKER_TOOL_INSTRUCTIONS}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Workspace: {self.workspace}\n"
            "Use the tools to read and write files to implement the task.\n"
        )
        return await self.client.run_tool_loop(
            prompt=prompt,
            execute=self._execute_tool,
            system=WORKER_PROMPT,
            model=self.client.settings.worker_model,
            tools=TOOLS,
            max_rounds=MAX_TOOL_ROUNDS,
        )
