from __future__ import annotations

import json
import re
from typing import Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

MAX_PLAN_ATTEMPTS = 3
MAX_RESPONSE_EXCERPT = 500

STRICT_JSON_INSTRUCTION = (
    " Respond with JSON only. Do not wrap it in markdown fences (```json ... ```), "
    "do not add prose before or after it, and do not add comments. "
    "If the goal is already complete, return an empty tasks array: "
    '{"tasks": [], "rationale": "Goal already complete."}'
)

# Scores used to choose between several balanced JSON spans in one reply: a
# plan-shaped object wins, a bare task list comes next, everything else ties and
# keeps the earliest span, which is the precedence the single-candidate scan had.
_RANK_PLAN_OBJECT = 2
_RANK_TASK_LIST = 1
_RANK_OTHER = 0


def _strip_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def extract_json(text: str) -> Any:
    candidates = [text, _strip_fence(text)]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        best: Any = None
        best_rank = -1
        for start, end in _find_outer(candidate):
            try:
                parsed = json.loads(candidate[start:end])
            except json.JSONDecodeError:
                # Balanced delimiters but not JSON (prose like ``[this]``).
                continue
            rank = _plan_rank(parsed)
            if rank > best_rank:
                best, best_rank = parsed, rank
            if best_rank == _RANK_PLAN_OBJECT:
                break
        if best_rank >= 0:
            return best
    raise ValueError("No parseable JSON found in LLM response")


def _plan_rank(value: Any) -> int:
    """Score a parsed span so a plan-shaped document outranks a prose list."""
    if isinstance(value, dict):
        if "tasks" in value or "plan" in value:
            return _RANK_PLAN_OBJECT
        return _RANK_OTHER
    if isinstance(value, list) and any(isinstance(item, dict) for item in value):
        return _RANK_TASK_LIST
    return _RANK_OTHER


def _find_outer(text: str) -> list[tuple[int, int]]:
    """Return the spans of every balanced ``{...}``/``[...]`` run in ``text``.

    String state is tracked from the first character of the text, not from the
    first opener, so a brace or bracket inside a quoted phrase of the surrounding
    prose is never mistaken for the start of the JSON document. An opener whose
    run never closes is abandoned in favour of the next one further along.
    """
    openers = "{["
    closers = "}]"
    spans: list[tuple[int, int]] = []
    stack: list[str] = []
    start = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in openers:
            if not stack:
                start = i
            stack.append(ch)
        elif ch in closers and stack:
            if closers[openers.index(stack.pop())] != ch:
                # Mismatched nesting is prose, not JSON: abandon the whole run.
                stack.clear()
            elif not stack:
                spans.append((start, i + 1))
    return spans


def _normalize_task(task: Any, index: int) -> TaskModel | None:
    if not isinstance(task, dict):
        return None
    tid = task.get("id")
    if not isinstance(tid, str) or not tid.strip():
        tid = str(index)
    desc = task.get("description")
    if not isinstance(desc, str) or not desc.strip():
        desc = "No description provided"
    files = task.get("files")
    if not isinstance(files, list):
        files = []
    deps = task.get("dependencies")
    if not isinstance(deps, list):
        deps = []
    # The single place where a normalized mapping becomes a real task object:
    # `Plan.tasks` is typed `list[TaskModel]`, and the values built above are
    # already exactly the declared types (`str`, `list[str]`), so constructing
    # the model here yields the same object `Plan(tasks=[...])` would have
    # produced for it.
    return TaskModel(
        id=tid,
        description=desc,
        files=[str(f) for f in files],
        dependencies=[str(d) for d in deps],
    )


def _build_plan(data: Any) -> Plan:
    if isinstance(data, list):
        data = {"tasks": data, "rationale": "Plan generated from task list."}
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object or array of tasks")
    if "plan" in data and isinstance(data["plan"], dict):
        data = data["plan"]
    tasks_raw = data.get("tasks")
    if not isinstance(tasks_raw, list):
        tasks_raw = []
    tasks: list[TaskModel] = []
    for i, raw in enumerate(tasks_raw):
        normalized = _normalize_task(raw, i)
        if normalized is None:
            continue
        tasks.append(normalized)
    rationale = data.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = "No rationale provided."
    return Plan(tasks=tasks, rationale=rationale)


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Any = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}{STRICT_JSON_INSTRUCTION}\n\nGoal: {goal}\n"
        last_error = ""
        for _attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
            response = await self.client.complete(
                prompt, model=self.client.settings.planner_model
            )
            try:
                data = extract_json(response)
                return _build_plan(data)
            except ValueError as e:
                last_error = str(e)
                excerpt = response[:MAX_RESPONSE_EXCERPT]
                prompt = (
                    f"{PLANNER_PROMPT}{STRICT_JSON_INSTRUCTION}\n\n"
                    f"Goal: {goal}\n\n"
                    f"Your previous response could not be parsed: {last_error}\n"
                    f"Previous response (truncated):\n{excerpt}\n\n"
                    "Respond again with JSON only, no fences, no prose."
                )
        raise ValueError(
            f"Failed to parse plan from LLM for goal: {goal!r}. Last error: {last_error}"
        )