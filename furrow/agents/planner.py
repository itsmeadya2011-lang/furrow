from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

#: Hard ceiling on how many tasks a single plan may contain.
MAX_TASKS = 5

#: The only task keys that are forwarded to :class:`~furrow.config.TaskModel`.
TASK_KEYS = ("id", "description", "files", "dependencies")

_FENCE_RE = re.compile(r"```[^\n`]*\n?(.*?)(?:```|\Z)", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Return the contents of the fenced code blocks in *text*, if any."""
    blocks = [block.strip() for block in _FENCE_RE.findall(text)]
    blocks = [block for block in blocks if block]
    if blocks:
        return "\n".join(blocks)
    return text.strip()


def _slice_json_object(text: str) -> str | None:
    """Slice the first balanced ``{...}`` object out of *text*.

    A brace-depth scan is used so that braces appearing inside string literals
    (and trailing prose such as ``Hope this helps!}``) do not truncate the
    candidate payload.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(text: str) -> dict:
    """Extract a single JSON object from an arbitrary LLM response.

    Handles bare JSON, ```json fenced blocks and JSON wrapped in prose.
    Raises :class:`ValueError` (including the raw response) when nothing parses.
    """
    raw = text if isinstance(text, str) else str(text)
    candidates = []
    stripped = _strip_code_fences(raw)
    if stripped:
        candidates.append(stripped)
    if raw not in candidates:
        candidates.append(raw)

    for candidate in candidates:
        blob = _slice_json_object(candidate)
        if blob is None:
            continue
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"No JSON object could be parsed from response: {raw}")


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _coerce_tasks(raw_tasks: Any) -> list[TaskModel]:
    """Drop unknown/invalid keys and unusable tasks from a raw task list."""
    if not isinstance(raw_tasks, list):
        return []
    tasks: list[TaskModel] = []
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            continue
        # Only the TASK_KEYS survive; everything else (including any LLM-invented
        # ``status``/``result``) is dropped before the model is built.
        task: dict[str, Any] = {}
        if isinstance(raw_task.get("id"), str) and raw_task["id"].strip():
            task["id"] = raw_task["id"]
        if isinstance(raw_task.get("description"), str) and raw_task["description"].strip():
            task["description"] = raw_task["description"]
        if "id" not in task or "description" not in task:
            continue
        if "files" in raw_task:
            task["files"] = _string_list(raw_task["files"])
        if "dependencies" in raw_task:
            task["dependencies"] = _string_list(raw_task["dependencies"])
        tasks.append(TaskModel(**task))
    return tasks


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    @property
    def max_tasks(self) -> int:
        try:
            configured = int(self.client.settings.max_parallel_tasks)
        except (TypeError, ValueError):
            configured = MAX_TASKS
        return max(0, min(MAX_TASKS, configured))

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = extract_json_object(response)
        except ValueError as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}") from e

        limit = self.max_tasks
        tasks = _coerce_tasks(data.get("tasks"))[:limit]
        if not tasks:
            raise ValueError(
                "Failed to parse plan from LLM: no usable tasks found"
                f" (max allowed: {limit}).\nResponse: {response}"
            )
        rationale = data.get("rationale")
        if not isinstance(rationale, str):
            rationale = "" if rationale is None else str(rationale)
        return Plan(tasks=tasks, rationale=rationale)