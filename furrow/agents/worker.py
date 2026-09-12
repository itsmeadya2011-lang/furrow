from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel, settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


MAX_TURNS = 5


class WorkerAgent:
    def __init__(
        self,
        task: TaskModel,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.task = task
        self.settings = settings or getattr(client, "settings", None) or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def run(self) -> str:
        workspace = self.settings.workspace.resolve()
        prompt = (
            f"{WORKER_PROMPT}\n\n"
            f"Task: {self.task.description}\n"
            f"Files to touch: {', '.join(self.task.files) if self.task.files else 'any'}\n"
            f"Workspace: {workspace}\n"
        )
        context: list[dict[str, str]] = [{"role": "user", "content": prompt}]

        for _ in range(MAX_TURNS):
            response = await self.client.complete(
                self._format_context(context),
                model=self.settings.worker_model,
            )
            if not isinstance(response, str):
                response = str(response)

            action = self._parse_action(response)
            if action is None:
                return response.strip()

            context.append({"role": "assistant", "content": response})
            action_error = self._validate_action(action)
            if action_error is not None:
                result = self._error_result(action, action_error)
                context.append({"role": "tool", "content": json.dumps(result)})
                continue

            if action["action"] == "done":
                return str(action["summary"])

            result = await self._execute_action(action, workspace)
            context.append({"role": "tool", "content": json.dumps(result)})

        return f"Stopped after {MAX_TURNS} model turns without a done action."

    @staticmethod
    def _format_context(context: list[dict[str, str]]) -> str:
        return "\n\n".join(
            f"{entry['role']}: {entry['content']}" for entry in context
        )

    @staticmethod
    def _parse_action(response: str) -> dict[str, Any] | None:
        text = response.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data

        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced is not None:
            try:
                data = json.loads(fenced.group(1))
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                return data

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                return data
        return None

    def _resolve_path(self, value: str, workspace: Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = workspace / path
        resolved = path.resolve()
        if not resolved.is_relative_to(workspace):
            raise ValueError("path is outside the workspace")
        return resolved

    def _validate_action(self, action: dict[str, Any]) -> str | None:
        action_name = action.get("action")
        expected_keys = {
            "read_file": {"action", "path"},
            "write_file": {"action", "path", "content"},
            "list_files": {"action", "directory"},
            "done": {"action", "summary"},
        }
        if not isinstance(action_name, str) or action_name not in expected_keys:
            return "unknown action"
        if set(action) != expected_keys[action_name]:
            return "action has an invalid shape"
        if action_name in {"read_file", "write_file"} and not isinstance(
            action["path"], str
        ):
            return "path must be a string"
        if action_name == "write_file" and not isinstance(action["content"], str):
            return "content must be a string"
        if action_name == "list_files" and not isinstance(action["directory"], str):
            return "directory must be a string"
        if action_name == "done" and not isinstance(action["summary"], str):
            return "summary must be a string"
        return None

    async def _execute_action(
        self, action: dict[str, Any], workspace: Path
    ) -> dict[str, Any]:
        action_name = action["action"]
        try:
            if action_name in {"read_file", "write_file"}:
                path = self._resolve_path(action["path"], workspace)
                if action_name == "read_file":
                    content = await self._maybe_await(
                        self.client.read_file(path)
                    )
                    if not isinstance(content, str):
                        raise TypeError("read_file must return text")
                    return {
                        "action": action_name,
                        "path": action["path"],
                        "status": "success",
                        "ok": True,
                        "content": content,
                    }

                await self._maybe_await(
                    self.client.write_file(path, action["content"])
                )
                return {
                    "action": action_name,
                    "path": action["path"],
                    "status": "success",
                    "ok": True,
                }

            directory = self._resolve_path(action["directory"], workspace)
            files = await self._maybe_await(self.client.list_files(directory))
            if not isinstance(files, list):
                raise TypeError("list_files must return a list")
            return {
                "action": action_name,
                "directory": action["directory"],
                "status": "success",
                "ok": True,
                "files": files,
            }
        except Exception as exc:
            return self._error_result(action, str(exc))

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _error_result(action: dict[str, Any], error: str) -> dict[str, Any]:
        return {
            "action": action.get("action"),
            "status": "error",
            "ok": False,
            "error": error,
        }
