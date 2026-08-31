from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents._json import _extract_json
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or Path.cwd()

    async def run(self) -> str:
        context_parts = [WORKER_PROMPT, f"\nTask: {self.task.description}\n"]
        if self.task.files:
            for rel in self.task.files:
                target = (self.workspace / rel).resolve()
                try:
                    existing = await self.client.read_file(target)
                    context_parts.append(f"### File: {rel}\n{existing}\n")
                except Exception:
                    context_parts.append(f"### File: {rel}\n(empty or missing)\n")
        else:
            try:
                listing = await asyncio_wrap(self.client.list_files, self.workspace)
                if listing:
                    context_parts.append("Existing files in workspace:\n" + "\n".join(f"  - {f}" for f in listing[:200]) + "\n")
            except Exception:
                pass

        prompt = "".join(context_parts)
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)
        try:
            data = json.loads(_extract_json(response))
        except (json.JSONDecodeError, ValueError):
            return response

        summary = data.get("summary", "")
        edits = data.get("edits", [])
        for edit in edits:
            rel = edit.get("path", "")
            if not rel:
                continue
            target = (self.workspace / rel).resolve()
            if edit.get("delete"):
                target.unlink(missing_ok=True)
            else:
                content = edit.get("content", "")
                target.parent.mkdir(parents=True, exist_ok=True)
                await self.client.write_file(target, content)
        return summary or "(no summary provided)"


async def asyncio_wrap(func, *args, **kwargs):
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
