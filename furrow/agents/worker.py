from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


class WorkerAgent:
    def __init__(self, task: TaskModel, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.task = task
        self.client = client or LLMClient(settings=settings)

    async def run(self) -> str:
        context_parts: list[str] = []
        for filepath in self.task.files:
            try:
                content = await self.client.read_file(filepath)
                context_parts.append(f"--- {filepath} ---\n{content}")
            except Exception:
                context_parts.append(f"--- {filepath} ---\n[file not found]")
        file_context = "\n\n".join(context_parts)

        prompt = f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n\nFiles to modify:\n{file_context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.worker_model)

        data = _extract_json(response)

        if data is None:
            text = response.strip()
            lowered = text.lower()
            if any(
                phrase in lowered
                for phrase in (
                    "no changes",
                    "did not change",
                    "unable to change",
                    "no change",
                    "nothing changed",
                    "nothing to change",
                    "no modifications",
                    "no edits",
                )
            ):
                raise RuntimeError(f"LLM indicated no changes: {text}")
            return text

        summary = str(data.get("summary", "Changes applied."))
        changes = data.get("changes", [])
        if not isinstance(changes, list):
            changes = []

        changed_paths: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        allowed_paths = set(self.task.files) if self.task.files else None

        for change in changes:
            if not isinstance(change, dict):
                continue
            action = change.get("action")
            path = change.get("path")
            if action not in ("write", "edit"):
                continue
            if not isinstance(path, str) or not path:
                continue

            if allowed_paths is not None and path not in allowed_paths:
                skipped.append(f"{path}: out of scope")
                continue

            try:
                if action == "write":
                    content = change.get("content", "")
                    if not isinstance(content, str) or not content:
                        skipped.append(f"{path}: missing content")
                        continue
                    await self.client.write_file(path, content)
                    changed_paths.append(path)
                elif action == "edit":
                    old_string = change.get("old_string", "")
                    new_string = change.get("new_string", "")
                    if not isinstance(old_string, str) or not isinstance(new_string, str):
                        skipped.append(f"{path}: invalid edit parameters")
                        continue
                    if not old_string:
                        skipped.append(f"{path}: empty old_string")
                        continue
                    current = await self.client.read_file(path)
                    if old_string not in current:
                        skipped.append(f"{path}: old_string not found")
                        continue
                    if current.count(old_string) > 1:
                        skipped.append(f"{path}: ambiguous edit ({current.count(old_string)} matches)")
                        continue
                    updated = current.replace(old_string, new_string, 1)
                    await self.client.write_file(path, updated)
                    changed_paths.append(path)
            except Exception as exc:
                failed.append(f"{path}: {exc}")
                continue

        if changed_paths:
            result = f"{summary}\nChanged: {', '.join(changed_paths)}"
            if skipped:
                result += f"\nSkipped: {'; '.join(skipped)}"
            if failed:
                raise RuntimeError(
                    f"Partial failure - applied changes to: {', '.join(changed_paths)}\n"
                    f"Failed: {'; '.join(failed)}"
                )
            return result
        if data is not None:
            details = "; ".join(skipped + failed) or "no valid changes"
            raise RuntimeError(f"LLM returned structured JSON but applied zero changes ({details})")
        parts = ["No changes were applied."]
        if skipped:
            parts.append(f"Skipped: {'; '.join(skipped)}")
        if failed:
            parts.append(f"Failed: {'; '.join(failed)}")
        return "\n".join(parts)
