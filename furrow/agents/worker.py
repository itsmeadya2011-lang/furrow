from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        files_block = "Files to touch: "
        files_block += ", ".join(self.task.files) if self.task.files else "any"
        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n{files_block}\n"

        existing_contents: list[str] = []
        if self.task.files:
            for path in self.task.files:
                try:
                    content = await self.client.read_file(path)
                except Exception:
                    continue
                existing_contents.append(f"--- {path} (existing) ---\n{content}")
        if existing_contents:
            prompt += "\nExisting file contents:\n" + "\n".join(existing_contents) + "\n"

        response = await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

        try:
            data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            return f"Could not parse structured response. Raw output:\n{response}"

        summary = data.get("summary", "")
        changes = data.get("changes", []) or []
        commands = data.get("commands", []) or []

        written: list[str] = []
        failed: list[str] = []
        for change in changes:
            path = change.get("path")
            content = change.get("content", "")
            if not path:
                continue
            try:
                await self.client.write_file(path, content)
                written.append(path)
            except Exception as exc:
                failed.append(f"{path} ({exc})")

        ran: list[str] = []
        for command in commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *command.split(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    ran.append(f"{command} (timed out after 120s)")
                    continue
                output = (stdout or b"").decode("utf-8", errors="replace")
                status = "ok" if proc.returncode == 0 else f"exit={proc.returncode}"
                ran.append(f"{command} [{status}]")
            except Exception as exc:
                ran.append(f"{command} (error: {exc})")

        parts = [summary]
        if written:
            parts.append("Wrote: " + ", ".join(written))
        if failed:
            parts.append("Failed: " + ", ".join(failed))
        if commands:
            parts.append("Ran: " + "; ".join(ran))
        return "\n".join(parts)
