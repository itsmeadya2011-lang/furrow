from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT, WORKER_WRITE_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Strip a single surrounding markdown code fence if present."""
    match = _FENCE_RE.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_actions(raw: str) -> list[dict[str, Any]]:
    """Parse the LLM JSON output into a list of action dicts."""
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise TypeError(f"Failed to parse LLM JSON output: {exc}") from exc

    if isinstance(data, dict):
        actions = data.get("actions")
    else:
        actions = data

    if not isinstance(actions, list):
        raise TypeError("Expected an 'actions' array in LLM output")

    normalized: list[dict[str, Any]] = []
    for item in actions:
        if not isinstance(item, dict):
            raise TypeError(f"Action must be an object, got {type(item).__name__}")
        action = item.get("action")
        if action not in ("read", "write"):
            raise ValueError(f"Unknown action {action!r}")
        if "path" not in item:
            raise ValueError(f"Action missing 'path': {item}")
        normalized.append(item)
    return normalized


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        workspace = self.client.settings.workspace
        files_list = ", ".join(self.task.files) if self.task.files else "any"

        # Phase 1: ask the LLM which files to read and which to write.
        plan_prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files_list}\n"
        )
        plan_raw = await self.client.complete(plan_prompt, model=self.client.settings.worker_model)
        try:
            plan_actions = _parse_actions(plan_raw)
        except TypeError as exc:
            return f"Failed to plan file operations: {exc}"

        # Execute reads (parallel where possible).
        read_paths = [a["path"] for a in plan_actions if a["action"] == "read"]
        file_contents: dict[str, str] = {}
        if read_paths:
            results = await asyncio.gather(
                *(self._safe_read(workspace / p) for p in read_paths),
                return_exceptions=True,
            )
            for path, result in zip(read_paths, results):
                if isinstance(result, Exception):
                    file_contents[path] = f"<error reading file: {result}>"
                else:
                    file_contents[path] = result

        # Phase 2: give the LLM the file contents and ask for write actions.
        contents_block = "\n\n".join(
            f"--- {path} ---\n{content}" for path, content in file_contents.items()
        ) or "(no files read)"

        write_prompt = WORKER_WRITE_PROMPT.format(
            task_description=self.task.description,
            files_list=files_list,
            file_contents=contents_block,
        )
        write_raw = await self.client.complete(write_prompt, model=self.client.settings.worker_model)
        try:
            write_actions = _parse_actions(write_raw)
        except TypeError as exc:
            return f"Failed to parse write actions: {exc}"

        # Execute writes.
        write_results: list[str] = []
        for action in write_actions:
            if action["action"] != "write":
                continue
            path = action["path"]
            content = action.get("content", "")
            try:
                await self.client.write_file(workspace / path, content)
                write_results.append(path)
            except OSError as exc:
                write_results.append(f"{path} (failed: {exc})")

        if not write_results:
            return "No write actions produced by the worker."

        summary = "Wrote files:\n" + "\n".join(f"- {p}" for p in write_results)
        if file_contents:
            summary += "\n\nRead files for context:\n" + "\n".join(
                f"- {p}" for p in file_contents
            )
        return summary

    async def _safe_read(self, path) -> str:
        try:
            return await self.client.read_file(path)
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface as a readable error
            return f"<error reading file: {exc}>"