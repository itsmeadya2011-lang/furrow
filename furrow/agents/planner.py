from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

MAX_PLAN_ATTEMPTS = 3
ERROR_ECHO_CHARS = 500

_OPENERS = {"{": "}", "[": "]"}


def _match_bracket(text: str, start: int) -> int | None:
    """Return the index just past the bracket closing the one at ``start``.

    String literals and escapes are respected so that braces inside JSON strings
    do not confuse the scan. Returns None when the span never closes.
    """
    stack: list[str] = []
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
        elif char in _OPENERS:
            stack.append(_OPENERS[char])
        elif char in ("}", "]"):
            if not stack or stack.pop() != char:
                return None
            if not stack:
                return index + 1
    return None


def _brace_spans(text: str) -> list[str]:
    """Return every well-formed {...} / [...] span, widest first."""
    spans: list[str] = []
    for index, char in enumerate(text):
        if char not in _OPENERS:
            continue
        end = _match_bracket(text, index)
        if end is not None:
            spans.append(text[index:end])
    spans.sort(key=len, reverse=True)
    return spans


def _fenced_body(text: str) -> str | None:
    """Return the body of the first ``` fence, or None when there is no fence.

    An unterminated fence (models forget the closing marker) yields everything
    after the opening line, which is usually still valid JSON.
    """
    start = text.find("```")
    if start == -1:
        return None
    line_end = text.find("\n", start)
    if line_end == -1:
        return text[start + 3 :]
    body_start = line_end + 1
    close = text.find("```", body_start)
    return text[body_start:] if close == -1 else text[body_start:close]


def extract_json(text: str) -> Any:
    """Best-effort extraction of a JSON value from an LLM response.

    Handles bare JSON, ```json / ``` fenced blocks, and JSON embedded in
    surrounding prose. Raises ValueError if nothing parses.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty LLM response contains no JSON")
    candidates: list[str] = []
    fenced = _fenced_body(stripped)
    if fenced and fenced.strip():
        candidates.append(fenced)
    candidates.append(stripped)
    candidates.extend(_brace_spans(stripped))
    for candidate in candidates:
        body = candidate.strip()
        if not body:
            continue
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON value found in LLM response: {text[:ERROR_ECHO_CHARS]!r}")


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _as_str_list(value: Any) -> list[str]:
    """Coerce None, a comma separated string, or a sequence into list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(";", ",").replace("\n", ",").split(",")
        return [part.strip() for part in parts if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [_as_text(item) for item in value if _as_text(item)]
    return []


def _task_description(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("description", "task", "title", "summary", "goal", "name"):
            text = _as_text(entry.get(key))
            if text:
                return text
    return ""


def _unique_id(raw: str, index: int, used: set[str]) -> str:
    base = raw.strip() or str(index)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def normalize_plan(data: Any) -> Plan:
    """Coerce a decoded LLM payload into a valid Plan, dropping junk entries.

    Accepts {"tasks": [...], "rationale": "..."}, a bare list of tasks, or a bare
    task string. Always raises ValueError when no usable task can be recovered.
    """
    rationale = ""
    raw_tasks: Any
    if isinstance(data, dict):
        rationale = _as_text(data.get("rationale")) or _as_text(data.get("reason"))
        raw_tasks = data.get("tasks")
        if raw_tasks is None:
            raw_tasks = data.get("steps")
        if raw_tasks is None:
            raw_tasks = data.get("plan")
    elif isinstance(data, list):
        raw_tasks = data
    else:
        raise ValueError(f"plan payload must be an object or list, got {type(data).__name__}")

    if isinstance(raw_tasks, (dict, str)):
        raw_tasks = [raw_tasks]
    if not isinstance(raw_tasks, list):
        raise ValueError(f"plan tasks must be a list, got {type(raw_tasks).__name__}")

    tasks: list[TaskModel] = []
    used_ids: set[str] = set()
    for index, entry in enumerate(raw_tasks, start=1):
        description = _task_description(entry)
        if not description:
            continue
        mapping = entry if isinstance(entry, dict) else {}
        task_id = _unique_id(_as_text(mapping.get("id")), index, used_ids)
        used_ids.add(task_id)
        tasks.append(
            TaskModel(
                id=task_id,
                description=description,
                files=_as_str_list(mapping.get("files")),
                dependencies=_as_str_list(mapping.get("dependencies")),
            )
        )
    if not tasks:
        raise ValueError("plan payload contained no usable task entries")

    known = {task.id for task in tasks}
    for task in tasks:
        task.dependencies = [
            dep for dep in dict.fromkeys(task.dependencies) if dep in known and dep != task.id
        ]
    if not rationale:
        rationale = "Plan of " + ", ".join(task.id for task in tasks) + "."
    return Plan(tasks=tasks, rationale=rationale)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        model = getattr(self.client.settings, "planner_model", None)
        prompt = f"Goal: {goal}\n\nProduce the task plan now, as bare JSON only."
        last_error = "no attempt was made"
        last_response = ""
        for attempt in range(MAX_PLAN_ATTEMPTS):
            response = await self.client.complete(prompt, system=PLANNER_PROMPT, model=model)
            last_response = response
            try:
                return normalize_plan(extract_json(response))
            except ValueError as exc:
                last_error = str(exc)
                if attempt == MAX_PLAN_ATTEMPTS - 1:
                    break
                prompt = self._retry_prompt(goal, last_error, response)
        raise ValueError(
            f"Failed to parse a usable plan after {MAX_PLAN_ATTEMPTS} attempts: {last_error}\n"
            f"Last response: {last_response[:ERROR_ECHO_CHARS]!r}"
        )

    @staticmethod
    def _retry_prompt(goal: str, error: str, response: str) -> str:
        return (
            f"Goal: {goal}\n\n"
            f"Your previous reply could not be used: {error}\n"
            f"Previous reply (truncated): {response[:ERROR_ECHO_CHARS]!r}\n\n"
            "Reply again with the raw JSON object only: no markdown fences, no prose, "
            'no comments. It must start with "{" and end with "}".'
        )
