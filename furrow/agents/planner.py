from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)

# Hard ceiling on a plan. Settings.max_parallel_tasks may lower it but never raise
# it: the runner starts one worker per task in parallel and the prompt promises a
# small slice, so an unbounded list would fan out unpredictably.
MAX_TASKS = 5
# The workspace listing is grounding only, so it stays small: paths, never file
# contents, or it would crowd the goal out of the prompt on every cycle.
MAX_LISTED_FILES = 60
# The previous cycle's failures are also grounding; a long list is summarized.
MAX_LISTED_FAILURES = 10
# One initial call plus one repair. Enough to recover from a fenced or chatty
# reply without letting a badly behaving model loop the orchestrator.
MAX_ATTEMPTS = 2
# Responses can be long; only this much is echoed into errors and repair prompts.
SNIPPET_CHARS = 600

# A fenced block, with or without a language tag. Requiring the closing fence means
# an unterminated fence is left alone and handled by the balanced-region scan.
_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)\n?```", re.DOTALL)

_BRACKET_PAIRS = {"{": "}", "[": "]"}
_OPENER_FOR_CLOSER = {"}": "{", "]": "["}

_REPAIR_TEMPLATE = """Your previous reply could not be parsed as JSON, so it was discarded.

Original goal:
{goal}

Your previous reply, verbatim:
---
{response}
---

Reply with raw JSON only: the first character must be {{ and the last must be }}.
No markdown fences, no ```json block, no prose before or after, no trailing commas.
Use exactly this shape:
{{"tasks": [{{"id": "1", "description": "...", "files": [], "dependencies": []}}],
 "rationale": "..."}}
"""


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Trim a response to something safe to log or paste into a prompt."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _strip_code_fences(text: str) -> str:
    """Return the body of the first fenced block, or the text unchanged.

    A fence is framing around the answer, so unwrapping it never loses content. An
    unfenced reply is left alone so the balanced-region scan can still find it.
    """
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _json_regions(text: str) -> list[str]:
    """Return every outermost balanced ``{...}``/``[...]`` span in ``text``.

    A stack scan instead of a regex is what makes surrounding prose safe: nested
    brackets stay inside the region and a brace that appears inside a JSON string
    does not move the depth. Every disjoint region is returned, not just the first,
    because a chatty reply often shows an example JSON object *before* the plan
    ('{"a": 1} is not a plan, use this:') and a single-region scan would return that
    decoy and never look at the answer that follows it.
    """
    regions: list[str] = []
    stack: list[str] = []
    start = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text):
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
        elif char in _BRACKET_PAIRS:
            if not stack:
                start = index
            stack.append(char)
        elif char in "}]" and stack:
            # On a mismatched pair, close through the nearest matching opener instead of
            # letting the stray bracket glue the rest of the reply into one region: that
            # would hide a valid plan sitting after the malformed part. This only decides
            # which text is *offered* to json.loads, which stays the sole authority on
            # whether it is valid.
            opener = _OPENER_FOR_CLOSER[char]
            while stack and stack[-1] != opener:
                stack.pop()
            if stack:
                stack.pop()
            if not stack:
                regions.append(text[start : index + 1])
    return regions


def _is_plan_shaped(payload: Any) -> bool:
    """Whether a payload could plausibly be the plan, used to skip decoy JSON."""
    if isinstance(payload, dict):
        return "tasks" in payload
    return isinstance(payload, list) and bool(payload)


def _is_non_empty(payload: Any) -> bool:
    """Whether a parsed payload carries content worth handing to the validator.

    Empty containers are treated as content-free because returning one would be a
    silent repair: a reply of ``{"tasks": [],}`` must fail loudly on its trailing
    comma rather than resolve to the empty ``[]`` hiding inside it.
    """
    if payload is None:
        return False
    if isinstance(payload, (dict, list, tuple, set, str, bytes)):
        return bool(payload)
    return True


def _payload_candidates(text: str) -> list[str]:
    """Every substring of ``text`` that could be the reply's JSON, in priority order.

    The unfenced body comes first because a fenced reply is pure JSON plus framing;
    its regions follow, so a plan wrapped in prose is still found.
    """
    unfenced = _strip_code_fences(text)
    candidates: list[str] = []
    for candidate in (unfenced, *_json_regions(unfenced)):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def extract_json_payload(text: str) -> Any:
    """Parse the JSON a model returned, tolerating framing but not invalid JSON.

    Models routinely fence the answer in ```json or wrap it in a sentence ("Here is
    the plan:"). Both are framing rather than content, so we drop the fences and then
    consider every outermost balanced object or array in turn, preferring the first
    one that looks like a plan (an object with a ``"tasks"`` key, or a non-empty
    array) so a decoy example earlier in the reply cannot win. Everything inside a
    region must still be valid JSON: trailing commas and single quotes are
    deliberately *not* repaired, because a silently reinterpreted plan is worse than
    one loud failure.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("planner returned an empty response")

    problems: list[str] = []
    fallback: Any = None
    have_fallback = False

    for candidate in _payload_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:  # try the next framing before giving up
            problems.append(f"{exc.msg} (line {exc.lineno} col {exc.colno})")
            continue
        if _is_plan_shaped(payload):
            return payload
        if not have_fallback and _is_non_empty(payload):
            fallback, have_fallback = payload, True

    if have_fallback:
        return fallback

    reason = "; ".join(problems) or "only empty JSON objects or arrays were found"
    raise ValueError(
        f"no parseable JSON in planner response ({reason}); "
        f"raw JSON only is required, response was: {_snippet(text)!r}"
    )


def _as_str_list(value: Any) -> list[str]:
    """Coerce an LLM field into a list of strings, defaulting to an empty list.

    Models emit a bare string, a null or a stray scalar where an array belongs; a
    best-effort coercion keeps the usable tasks instead of failing the whole cycle.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        return [
            str(item).strip()
            for item in value
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]
    return []


def _task_cap(max_tasks: int) -> int:
    """Resolve how many tasks a plan may contain from ``max_parallel_tasks``.

    The floor of 1 avoids a misconfigured 0 turning into an empty plan, which the
    runner reads as "goal complete" and halts on. The ceiling of ``MAX_TASKS`` is
    the hard cap stated in the planner prompt.
    """
    try:
        requested = int(max_tasks)
    except (TypeError, ValueError):
        requested = MAX_TASKS
    return max(1, min(MAX_TASKS, requested))


def _normalize_tasks(raw_tasks: Any) -> list[TaskModel]:
    """Coerce the model's task list into ``TaskModel`` objects, dropping bad entries.

    A single malformed entry (no description, a string instead of an object, ...)
    should cost one task, not the whole cycle, so unusable entries are logged and
    skipped. Ids are made unique and non-empty because they are the join key for
    dependencies; a missing or duplicated id would make those references ambiguous.
    """
    if not isinstance(raw_tasks, list):
        raise ValueError(f"'tasks' must be a JSON array, got {type(raw_tasks).__name__}")
    if not raw_tasks:
        raise ValueError("'tasks' is empty; a plan must contain at least one task")

    tasks: list[TaskModel] = []
    used_ids: set[str] = set()
    next_id = 1

    for index, entry in enumerate(raw_tasks):
        if not isinstance(entry, dict):
            logger.warning("Dropping plan task %s: not a JSON object (%r)", index, entry)
            continue
        data = {key: value for key, value in entry.items() if key in TaskModel.model_fields}

        raw_id = data.get("id")
        task_id = str(raw_id).strip() if isinstance(raw_id, (str, int, float)) else ""
        if not task_id or task_id in used_ids:
            candidate = str(next_id)
            while candidate in used_ids:
                next_id += 1
                candidate = str(next_id)
            logger.warning(
                "Renaming plan task %s: id %r is missing or duplicated, using %r",
                index,
                raw_id,
                candidate,
            )
            task_id = candidate
        if task_id.isdigit():
            next_id = max(next_id, int(task_id) + 1)
        used_ids.add(task_id)

        data["id"] = task_id
        data["files"] = _as_str_list(data.get("files"))
        data["dependencies"] = _as_str_list(data.get("dependencies"))
        # The runner owns lifecycle state; a model-supplied "completed" status would
        # otherwise make the orchestrator think the goal is already met.
        data["status"] = "pending"
        data["result"] = None
        try:
            tasks.append(TaskModel(**data))
        except ValidationError as exc:
            logger.warning("Dropping plan task %s (id %r): %s", index, task_id, exc)

    if not tasks:
        raise ValueError("no usable tasks: every entry in 'tasks' failed validation")
    return tasks


def _filter_dependencies(tasks: list[TaskModel]) -> None:
    """Drop duplicate, unknown and self-referencing dependency ids.

    A dependency on an id that is not in the plan would leave a task waiting for
    work nobody runs, which stalls the scheduler rather than failing fast.
    """
    known = {task.id for task in tasks}
    for task in tasks:
        kept = [
            dep for dep in dict.fromkeys(task.dependencies) if dep in known and dep != task.id
        ]
        dropped = [dep for dep in task.dependencies if dep not in kept]
        if dropped:
            logger.warning("Task %s: dropping unknown dependencies %s", task.id, dropped)
        task.dependencies = kept


def _break_dependency_cycles(tasks: list[TaskModel]) -> None:
    """Remove back-edges so the task graph is acyclic.

    The runner only starts a task once its dependencies finish, so any cycle would
    wait forever. An iterative DFS that deletes every edge pointing back into the
    path currently being walked is the minimal fix: no cycle can survive it, and it
    cannot loop because every edge is consumed once.
    """
    by_id = {task.id: task for task in tasks}
    # 0 = on the path being walked, 1 = finished, absent = not reached yet.
    state: dict[str, int] = {}

    for start in tasks:
        if start.id in state:
            continue
        state[start.id] = 0
        stack: list[tuple[str, list[str]]] = [(start.id, list(start.dependencies))]
        while stack:
            node_id, pending = stack[-1]
            if not pending:
                state[node_id] = 1
                stack.pop()
                continue
            dep_id = pending.pop(0)
            if dep_id not in by_id:
                continue
            if state.get(dep_id, 1) == 0:
                by_id[node_id].dependencies = [
                    dep for dep in by_id[node_id].dependencies if dep != dep_id
                ]
                logger.warning(
                    "Task %s: dropped dependency %s because it forms a cycle",
                    node_id,
                    dep_id,
                )
                continue
            if dep_id not in state:
                state[dep_id] = 0
                stack.append((dep_id, list(by_id[dep_id].dependencies)))


def build_plan(payload: Any, max_tasks: int) -> Plan:
    """Turn a parsed payload into a validated, normalized, schedulable ``Plan``."""
    if isinstance(payload, list):
        raw_tasks: Any = payload  # a bare array of tasks is accepted as the plan
        rationale = ""
    elif isinstance(payload, dict):
        raw_tasks = payload.get("tasks")
        raw_rationale = payload.get("rationale")
        rationale = str(raw_rationale) if isinstance(raw_rationale, str) else ""
    else:
        raise ValueError(
            f"plan must be a JSON object with a 'tasks' array, got {type(payload).__name__}"
        )

    tasks = _normalize_tasks(raw_tasks)
    cap = _task_cap(max_tasks)
    if len(tasks) > cap:
        # The prompt asks for at most 5; keep the first N in the order the model
        # listed them, which is its own priority order, and say so in the rationale.
        logger.warning("Clamping plan from %s to %s tasks", len(tasks), cap)
        tasks = tasks[:cap]
        rationale = (
            f"{rationale} Clamped to the first {cap} of the {len(tasks)} planned tasks "
            f"(limit: max_parallel_tasks={max_tasks}).".strip()
        )

    _filter_dependencies(tasks)
    _break_dependency_cycles(tasks)
    return Plan(tasks=tasks, rationale=rationale)


class PlannerAgent:
    """Turns a goal into a validated, dependency-safe :class:`Plan`."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client if client is not None else LLMClient(settings=settings)
        # The orchestrator injects a client built from its own Settings, so read the
        # model name, the task cap and the workspace off the client rather than off
        # the module-level default, which may describe a different configuration.
        self.settings: Settings = getattr(self.client, "settings", None) or settings or Settings()

    async def plan(
        self, goal: str, test_failures: Sequence[str] | None = None
    ) -> Plan:
        """Plan one cycle, repairing malformed output once before giving up.

        ``test_failures`` is what the previous cycle's tester found; when present
        the plan is steered at fixing those rather than re-planning the whole goal.
        """
        prompt = self._build_prompt(goal, test_failures)
        responses: list[str] = []
        last_error: ValueError | None = None

        for attempt in range(MAX_ATTEMPTS):
            response = await self.client.complete(prompt, model=self.settings.planner_model)
            responses.append(response)
            try:
                return build_plan(
                    extract_json_payload(response), self.settings.max_parallel_tasks
                )
            except ValueError as exc:
                last_error = exc
                if attempt + 1 < MAX_ATTEMPTS:
                    logger.warning("Planner output unusable (%s); requesting a repair", exc)
                    prompt = self._build_repair_prompt(goal, response)

        raise ValueError(
            f"Planner could not return a usable plan after {MAX_ATTEMPTS} attempts: "
            f"{last_error}\nGoal: {_snippet(goal, 200)}\nResponses:\n"
            + "\n".join(f"- {_snippet(response, 300)}" for response in responses)
        ) from last_error

    def _build_prompt(self, goal: str, test_failures: Sequence[str] | None = None) -> str:
        sections = [PLANNER_PROMPT, f"\nGoal: {goal}"]
        listing = self._file_listing()
        if listing:
            sections.append(
                "\nWorkspace files (partial listing of paths only; a worker reads a file "
                "before editing it):\n" + "\n".join(f"- {path}" for path in listing)
            )
        failures = [str(f).strip() for f in (test_failures or []) if str(f).strip()]
        if failures:
            sections.append(
                "\nThe previous cycle left these test failures. Plan the fixes for them, "
                "not the whole goal again:\n"
                + "\n".join(f"- {failure[:300]}" for failure in failures[:MAX_LISTED_FAILURES])
            )
        return "\n".join(sections)

    def _file_listing(self) -> list[str]:
        """Best-effort bounded workspace listing used as prompt grounding.

        Paths only, never contents, to keep the prompt small. A listing failure is
        never fatal: it just means this cycle is planned without workspace context.
        """
        try:
            paths = [str(path) for path in self.client.list_files(self.settings.workspace)]
        except Exception:  # grounding is best-effort, never a reason to fail a cycle
            logger.warning("Could not list workspace files", exc_info=True)
            return []
        return sorted(path for path in paths if path)[:MAX_LISTED_FILES]

    def _build_repair_prompt(self, goal: str, bad_response: str) -> str:
        # Only the goal and the bad reply: the repair call exists to fix the syntax,
        # and re-sending the workspace listing would just cost tokens.
        return _REPAIR_TEMPLATE.format(
            goal=_snippet(goal, 500), response=_snippet(bad_response, 1200)
        )
