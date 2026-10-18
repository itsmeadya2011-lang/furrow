from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)
        self.read_file = self.client.read_file
        self.write_file = self.client.write_file
        self.list_files = self.client.list_files

    async def _complete(self, prompt: str) -> str:
        return await self.client.complete(
            prompt, model=self.client.settings.worker_model
        )

    @staticmethod
    def _parse_actions(text: str) -> list[dict[str, Any]] | None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        data = json.loads(cleaned[start : end + 1])
                    except json.JSONDecodeError:
                        return None
                else:
                    return None
        actions = data.get("actions", []) if isinstance(data, dict) else []
        if not isinstance(actions, list):
            return None
        return actions

    async def _first_round(self) -> tuple[list[dict[str, Any]], str]:
        files = ", ".join(self.task.files) if self.task.files else "any"
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n"
            f"First, return read/list actions to inspect the relevant files. "
            f"Do not write anything yet.\n"
        )
        text = await self._complete(prompt)
        actions = self._parse_actions(text) or []
        return actions, text

    async def _second_round(self, file_contents: dict[str, str]) -> list[dict[str, Any]]:
        files = ", ".join(self.task.files) if self.task.files else "any"
        context = "\n\n".join(
            f"--- {path} ---\n{content}" for path, content in file_contents.items()
        )
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {files}\n"
            f"Current file contents:\n{context}\n\n"
            f"Now return the final write actions with the complete file contents "
            f"to implement the task.\n"
        )
        text = await self._complete(prompt)
        actions = self._parse_actions(text)
        return actions if actions is not None else []

    async def run(self) -> str:
        actions, raw = await self._first_round()

        file_contents: dict[str, str] = {}
        first_round_writes: list[dict[str, Any]] = []

        for action in actions:
            kind = action.get("action")
            path = action.get("path")
            if not path:
                continue
            if kind == "read":
                try:
                    file_contents[str(path)] = await self.read_file(path)
                except Exception:
                    pass
            elif kind == "list":
                try:
                    listing = await self.list_files(path)
                    file_contents[str(path)] = "\n".join(listing)
                except Exception:
                    pass
            elif kind == "write":
                first_round_writes.append(action)

        write_actions = first_round_writes
        if file_contents or not write_actions:
            second = await self._second_round(file_contents)
            second_writes = [
                action for action in second if action.get("action") == "write"
            ]
            if second_writes:
                write_actions = second_writes

        written: list[str] = []
        errors: list[str] = []
        for action in write_actions:
            path = action.get("path")
            content = action.get("content", "")
            if not path:
                continue
            try:
                await self.write_file(path, content)
                written.append(str(path))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{path}: {exc}")

        summary_parts: list[str] = []
        if written:
            summary_parts.append(f"Wrote {len(written)} file(s): {', '.join(written)}")
        if errors:
            summary_parts.append(f"Errors: {'; '.join(errors)}")
        if not summary_parts:
            if raw:
                return f"No actions produced. Raw LLM response: {raw[:200]}"
            return "No actions produced."
        return "; ".join(summary_parts)
