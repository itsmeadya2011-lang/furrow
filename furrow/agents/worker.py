from __future__ import annotations

import re
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
        self.errors: list[str] = []

    async def run(self) -> str:
        context_parts: list[str] = []
        if self.task.files:
            for f in self.task.files:
                try:
                    content = await self.client.read_file(f)
                    context_parts.append(f"--- {f} ---\n{content}")
                except Exception as exc:  # noqa: BLE001
                    self.errors.append(f"read {f}: {exc}")
                    context_parts.append(f"--- {f} ---\n<unreadable: {exc}>")

        conversation = (
            f"{WORKER_PROMPT}\n\nTask: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
        )
        if context_parts:
            conversation += "\nContext files:\n" + "\n".join(context_parts) + "\n"

        history: list[str] = [conversation]

        for _ in range(10):
            prompt = "\n\n".join(history)
            try:
                response = await self.client.complete(
                    prompt, model=self.client.settings.worker_model
                )
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"llm: {exc}")
                return self._summary(f"LLM error: {exc}")

            action = self._parse(response)
            if action is None:
                history.append(f"Assistant:\n{response}")
                history.append(
                    "User:\nNo valid action found. Issue exactly one action block."
                )
                continue

            kind, payload = action
            history.append(f"Assistant:\n{response}")

            if kind == "DONE":
                summary = payload.strip()
                if self.errors:
                    summary += "\nErrors: " + "; ".join(self.errors)
                return summary

            if kind == "READ":
                path = payload.strip()
                try:
                    content = await self.client.read_file(path)
                    result = f"File contents of {path}:\n{content}"
                except Exception as exc:  # noqa: BLE001
                    self.errors.append(f"read {path}: {exc}")
                    result = f"Error reading {path}: {exc}"
                history.append(f"User:\n{result}")

            elif kind == "WRITE":
                path, content = payload
                try:
                    p = Path(path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    await self.client.write_file(p, content)
                    result = f"Wrote {path} successfully."
                except Exception as exc:  # noqa: BLE001
                    self.errors.append(f"write {path}: {exc}")
                    result = f"Error writing {path}: {exc}"
                history.append(f"User:\n{result}")

            elif kind == "LIST":
                directory = payload.strip() or "."
                try:
                    files = self.client.list_files(directory)
                    result = f"Files in {directory}:\n" + "\n".join(files)
                except Exception as exc:  # noqa: BLE001
                    self.errors.append(f"list {directory}: {exc}")
                    result = f"Error listing {directory}: {exc}"
                history.append(f"User:\n{result}")

        msg = "Max iterations reached without DONE."
        if self.errors:
            msg += " Errors: " + "; ".join(self.errors)
        return msg

    def _parse(self, text: str):
        m = re.search(r"<<DONE>>(.*?)<<END>>", text, re.DOTALL)
        if m:
            return ("DONE", m.group(1))

        m = re.search(r"<<READ>>(.*?)<<END>>", text, re.DOTALL)
        if m:
            return ("READ", m.group(1))

        m = re.search(r"<<LIST>>(.*?)<<END>>", text, re.DOTALL)
        if m:
            return ("LIST", m.group(1))

        m = re.search(r"<<WRITE>>(.*?)<<END>>", text, re.DOTALL)
        if m:
            payload = m.group(1)
            if "\n" in payload:
                path, content = payload.split("\n", 1)
                if content.endswith("\n"):
                    content = content[:-1]
            else:
                path, content = payload, ""
            return ("WRITE", (path.strip(), content))

        return None

    def _summary(self, msg: str) -> str:
        if self.errors:
            msg += " Errors: " + "; ".join(self.errors)
        return msg
