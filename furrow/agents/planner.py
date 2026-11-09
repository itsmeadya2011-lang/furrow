from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient, ToolCall


PLANNER_TOOLS: list[dict] = [
    {
        "name": "analyze_codebase",
        "description": "Analyze the codebase structure to understand the project layout, key files, and patterns. Returns a summary of the project structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "max_depth": {"type": "integer", "default": 3},
                "include_patterns": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
        },
    },
]


class PlannerAgent:
    def __init__(
        self, client: LLMClient | None = None, settings: Settings | None = None, workspace: Path | None = None
    ) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self.workspace = Path(workspace) if workspace is not None else Path(self.settings.workspace)

    async def plan(self, goal: str, context: str = "") -> Plan:
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if context:
            prompt += "\n\nContext:\n" + context

        model = self.settings.planner_model
        messages = [{"role": "user", "content": prompt}]
        response = await self.client.chat(
            messages, tools=PLANNER_TOOLS, system=PLANNER_SYSTEM, model=model
        )

        # Execute any tool calls from the planner
        while response.tool_calls:
            blocks: list[dict[str, Any]] = []
            if response.text:
                blocks.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            messages.append({"role": "assistant", "content": blocks})

            # Execute tools
            result_blocks: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                result = await self._execute_tool(tc)
                result_blocks.append(
                    {"type": "tool_result", "tool_use_id": tc.id, "content": result}
                )
            messages.append({"role": "user", "content": result_blocks})

            # Get next response
            response = await self.client.chat(
                messages, tools=PLANNER_TOOLS, system=PLANNER_SYSTEM, model=model
            )

        # Now parse the final response as a plan
        try:
            return parse_plan(response.text)
        except ValueError as first:
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                return parse_plan(response2)
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second

    async def _execute_tool(self, tc: ToolCall) -> str:
        name = tc.name
        args = tc.arguments
        if name == "analyze_codebase":
            return await self._analyze_codebase(
                int(args.get("max_depth", 3)),
                args.get("include_patterns", None),
            )
        return f"unknown tool: {name}"

    async def _analyze_codebase(
        self, max_depth: int = 3, include_patterns: list[str] | None = None
    ) -> str:
        """Analyze the codebase structure and return a summary."""
        from pathlib import Path

        workspace = self.workspace.resolve()
        if not workspace.exists():
            return "Workspace does not exist"

        # Common patterns to ignore
        ignore_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".furrow",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "target",
            ".next",
            ".vercel",
        }

        def should_ignore(path: Path) -> bool:
            for part in path.parts:
                if part in ignore_dirs:
                    return True
            return False

        structure: dict[str, Any] = {"directories": [], "files": []}

        def walk(current: Path, depth: int = 0) -> None:
            if depth > max_depth:
                return
            if should_ignore(current):
                return

            try:
                entries = sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            except (OSError, PermissionError):
                return

            for entry in entries:
                if should_ignore(entry):
                    continue
                rel = entry.relative_to(workspace)
                if entry.is_dir():
                    structure["directories"].append(str(rel))
                    walk(entry, depth + 1)
                else:
                    # Check include patterns
                    if include_patterns:
                        import fnmatch
                        if not any(fnmatch.fnmatch(entry.name, pat) for pat in include_patterns):
                            continue
                    structure["files"].append(str(rel))

        walk(workspace)

        # Limit output size
        max_items = 200
        dirs = structure["directories"][:max_items]
        files = structure["files"][:max_items]

        summary = f"Project structure (max depth {max_depth}):\n"
        if dirs:
            summary += "\nDirectories:\n" + "\n".join(f"  {d}/" for d in dirs)
            if len(structure["directories"]) > max_items:
                summary += f"\n  ... ({len(structure['directories']) - max_items} more directories)"
        if files:
            summary += "\n\nFiles:\n" + "\n".join(f"  {f}" for f in files)
            if len(structure["files"]) > max_items:
                summary += f"\n  ... ({len(structure['files']) - max_items} more files)"

        return summary
