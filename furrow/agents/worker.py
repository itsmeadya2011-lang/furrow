from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    def _allowed_files(self) -> set[str]:
        files = set(self.task.files)
        if self.task.files:
            workspace = getattr(self.client.settings, "workspace", None)
            if workspace:
                files.update(str(Path(workspace) / f) for f in self.task.files)
        return files

    async def _run_bash(self, command: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        return (stdout or b"").decode("utf-8", errors="replace")

    async def _execute(self, action: dict) -> str:
        atype = action.get("type")
        if atype == "read":
            path = action["path"]
            try:
                return await self.client.read_file(path)
            except Exception as exc:
                return f"read_error: {exc}"
        elif atype == "write":
            path = action["path"]
            content = action.get("content", "")
            try:
                await self.client.write_file(path, content)
                return f"wrote {path}"
            except Exception as exc:
                return f"write_error: {exc}"
        elif atype == "bash":
            command = action["command"]
            try:
                return await self._run_bash(command)
            except Exception as exc:
                return f"bash_error: {exc}"
        else:
            return f"unknown_action_type: {atype}"

    async def run(self) -> str:
        allowed = self._allowed_files()
        files_str = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
            f"Files to touch: {files_str}\n"
        )
        model = self.client.settings.worker_model

        history: list[dict] = []
        final_summary = ""
        for _ in range(3):
            response = await self.client.complete(prompt, model=model)
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError as exc:
                return f"Failed to parse LLM JSON response: {exc}\nRaw: {response[:500]}"

            actions = parsed.get("actions", []) or []
            summary = parsed.get("summary", "")
            final_summary = summary

            if not actions:
                return summary if summary else "No actions required."

            for action in actions:
                path = action.get("path")
                if path and allowed and path not in allowed:
                    result = (
                        f"skipped: {path} is outside allowed files "
                        f"({', '.join(sorted(allowed))})"
                    )
                else:
                    result = await self._execute(action)
                history.append({"action": action, "result": result})

            results_block = "\n\n".join(
                f"Action: {json.dumps(h['action'])}\nResult: {h['result']}"
                for h in history
            )
            prompt = (
                f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
                f"Files to touch: {files_str}\n\n"
                f"You have already taken the following actions. Results follow.\n\n"
                f"{results_block}\n\n"
                f"Take the next actions needed to complete the task. "
                f"Return JSON only (no markdown, no explanation)."
            )

        return final_summary if final_summary else "Completed 3 rounds."

