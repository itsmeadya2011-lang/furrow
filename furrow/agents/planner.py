from __future__ import annotations

import json
from pathlib import PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


_ALLOWED_PLAN_FIELDS = {"tasks", "rationale"}
_ALLOWED_TASK_FIELDS = {"id", "description", "files", "dependencies", "status", "result"}
_ALLOWED_STATUSES = {"pending", "running", "completed", "failed", "blocked"}


def _response_text(response: object) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, bytes):
        return response.decode("utf-8", errors="replace")
    return str(response)


def _extract_json_object(response: object) -> str:
    text = _response_text(response).lstrip("﻿").strip()
    if not text:
        raise ValueError("response is empty")

    start = text.find("{")
    if start == -1:
        raise ValueError("response does not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("response contains an unbalanced JSON object")


def _relative_file_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("file paths must be non-empty strings")
    if "\x00" in value or any(ord(character) < 32 for character in value):
        raise ValueError("file paths must not contain control characters")

    normalized = value.replace("\\", "/")
    if (
        normalized.startswith(("/", "\\\\", "//"))
        or PureWindowsPath(value).drive
        or normalized.endswith("/")
        or normalized.endswith("/.")
    ):
        raise ValueError(f"file path must be relative to the workspace: {value!r}")

    path = PurePosixPath(normalized)
    if not path.parts or any(part == ".." for part in path.parts):
        raise ValueError(f"file path must stay within the workspace: {value!r}")
    return normalized


def _validate_plan_data(data: object) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("plan must be a JSON object")

    unknown_fields = set(data) - _ALLOWED_PLAN_FIELDS
    if unknown_fields:
        raise ValueError(f"plan contains unknown fields: {', '.join(sorted(unknown_fields))}")
    if "tasks" not in data or "rationale" not in data:
        raise ValueError("plan requires tasks and rationale fields")
    if not isinstance(data["rationale"], str) or not data["rationale"].strip():
        raise ValueError("plan rationale must be a non-empty string")
    if not isinstance(data["tasks"], list):
        raise ValueError("plan tasks must be an array")
    if len(data["tasks"]) > 5:
        raise ValueError("plan contains more than five tasks")

    tasks: list[dict[str, Any]] = []
    ids: list[str] = []
    graph: dict[str, list[str]] = {}
    for index, task in enumerate(data["tasks"]):
        if not isinstance(task, dict):
            raise ValueError(f"task {index} must be a JSON object")
        unknown_task_fields = set(task) - _ALLOWED_TASK_FIELDS
        if unknown_task_fields:
            raise ValueError(
                f"task {index} contains unknown fields: {', '.join(sorted(unknown_task_fields))}"
            )
        for field in ("id", "description"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise ValueError(f"task {index} {field} must be a non-empty string")

        files_value = task.get("files", [])
        dependencies_value = task.get("dependencies", [])
        if not isinstance(files_value, list) or not all(
            isinstance(item, str) and item.strip() for item in files_value
        ):
            raise ValueError(f"task {index} files must be an array of non-empty strings")
        if not isinstance(dependencies_value, list) or not all(
            isinstance(item, str) and item.strip() for item in dependencies_value
        ):
            raise ValueError(f"task {index} dependencies must be an array of non-empty strings")
        if len(dependencies_value) != len(set(dependencies_value)):
            raise ValueError(f"task {index} dependencies must be unique")

        task_id = task["id"].strip()
        description = task["description"].strip()
        files = [_relative_file_path(item) for item in files_value]
        dependencies = [dependency.strip() for dependency in dependencies_value]
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"task {task_id!r} dependencies must be unique")
        if task_id in dependencies:
            raise ValueError(f"task {task_id!r} cannot depend on itself")

        status = task.get("status", "pending")
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"task {task_id!r} has an invalid status")
        result = task.get("result")
        if result is not None and not isinstance(result, str):
            raise ValueError(f"task {task_id!r} result must be a string or null")

        ids.append(task_id)
        graph[task_id] = dependencies
        tasks.append(
            {
                "id": task_id,
                "description": description,
                "files": files,
                "dependencies": dependencies,
                "status": status,
                "result": result,
            }
        )

    if len(ids) != len(set(ids)):
        raise ValueError("plan task ids must be unique")

    known_ids = set(ids)
    for task in tasks:
        missing = [dependency for dependency in task["dependencies"] if dependency not in known_ids]
        if missing:
            raise ValueError(
                f"task {task['id']!r} has unknown dependencies: {', '.join(missing)}"
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(f"plan dependencies contain a cycle at task {task_id!r}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in graph[task_id]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in ids:
        visit(task_id)

    return {"tasks": tasks, "rationale": data["rationale"].strip()}


def _extract_json(response: object) -> str:
    return _extract_json_object(response)


def _validate_plan(data: object) -> dict[str, Any]:
    return _validate_plan_data(data)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings
        if self.settings is None:
            self.settings = getattr(client, "settings", None) or default_settings
        self.client = client or LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "", original_goal: str | None = None) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}"
        if original_goal:
            prompt += f"\n\nOriginal goal: {original_goal}"
        if context:
            prompt += f"\n\n{context}"

        max_attempts = 2
        try:
            max_attempts = max(1, int(getattr(self.settings, "max_attempts", 1)))
        except (TypeError, ValueError):
            pass

        last_response = ""
        last_error: ValueError | None = None
        for attempt in range(1, max_attempts + 1):
            response = await self.client.complete(
                prompt,
                model=getattr(self.settings, "planner_model", None)
                or getattr(getattr(self.client, "settings", None), "planner_model", None)
                or default_settings.planner_model,
            )
            last_response = _response_text(response)
            try:
                data = json.loads(_extract_json_object(last_response))
                return Plan(**_validate_plan_data(data))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt == max_attempts:
                    break

        detail = f": {last_error}" if last_error else ""
        raise ValueError(
            f"Failed to parse and validate plan from LLM after {max_attempts} attempt(s){detail}\n"
            f"Response: {last_response}"
        )
