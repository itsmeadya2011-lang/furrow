from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)

# PLANNER_PROMPT asks for 1-5 tasks. Whatever comes back is clamped to
# ``settings.max_parallel_tasks`` and never above this hard ceiling, so a
# hallucinated forty task plan cannot blow up the worker fan-out.
HARD_MAX_TASKS = 5

# Repository listing injected into the prompt: enough to ground file paths in
# reality without flooding the context window.
MAX_CONTEXT_FILES = 120
MAX_CONTEXT_CHARS = 6000

# Initial attempt plus one corrective retry. A single malformed reply must not
# abort the whole cycle, but we must not loop on a hopeless prompt either.
MAX_PLAN_ATTEMPTS = 2
MAX_ERROR_CHARS = 400

# ``Plan.rationale`` is required, so a reply that omits it gets a safe default
# instead of a validation crash.
DEFAULT_RATIONALE = "Plan derived from the goal and the repository listing; no rationale was returned."

CORRECTION_INSTRUCTION = (
    "The previous reply could not be parsed as JSON ({error}).\n"
    "Reply again with a single raw JSON object matching the required shape: "
    "no markdown fences, no preamble, no explanation, no trailing commentary."
)


class PlannerAgent:
    """Turns a goal into a validated, dependency-clean :class:`Plan`."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    # -------------------------------------------------------------------- plan

    async def plan(self, goal: str) -> Plan:
        """Return a sanitised plan for ``goal``.

        ``goal`` may already carry prior failure feedback (the orchestrator
        appends it), so it is forwarded verbatim as the current goal.
        """
        base_prompt = self._build_prompt(goal)
        prompt = base_prompt
        last_error = ""
        last_response = ""

        for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
            response = await self.client.complete(prompt, model=self.client.settings.planner_model)
            last_response = response

            try:
                data = json.loads(extract_json(response))
            except ValueError as exc:
                last_error = str(exc)
                logger.warning(
                    "Planner reply was not parseable JSON (attempt %d/%d): %s",
                    attempt,
                    MAX_PLAN_ATTEMPTS,
                    last_error,
                )
                prompt = self._corrective_prompt(base_prompt, last_error)
                continue

            if not isinstance(data, dict):
                last_error = f"expected a JSON object, got {type(data).__name__}"
                logger.warning(
                    "Planner reply was not a JSON object (attempt %d/%d): %s",
                    attempt,
                    MAX_PLAN_ATTEMPTS,
                    last_error,
                )
                prompt = self._corrective_prompt(base_prompt, last_error)
                continue

            return self._build_plan(data)

        response_excerpt, _ = _clip_text(last_response, MAX_ERROR_CHARS)
        raise ValueError(
            f"Failed to parse plan from LLM after {MAX_PLAN_ATTEMPTS} attempts: {last_error}\n"
            f"Response: {response_excerpt}"
        )

    @staticmethod
    def _corrective_prompt(base_prompt: str, error: str) -> str:
        excerpt, _ = _clip_text(error, MAX_ERROR_CHARS)
        return f"{base_prompt}\n\n{CORRECTION_INSTRUCTION.format(error=excerpt)}"

    # ------------------------------------------------------------------ prompt

    def _build_prompt(self, goal: str) -> str:
        sections = [PLANNER_PROMPT, f"Goal: {goal}", self._repository_context()]
        return "\n\n".join(section for section in sections if section)

    def _repository_context(self) -> str:
        """Ground the plan in the real repository listing."""
        workspace = getattr(self.client.settings, "workspace", None) or "."
        try:
            files = self.client.list_files(workspace, limit=MAX_CONTEXT_FILES)
        except OSError as exc:  # unreadable workspace must not abort planning
            logger.warning("Could not list %s for planning context: %s", workspace, exc)
            return ""

        if not files:
            return f"Workspace: {workspace}\n(no files found)"

        listing, clipped = _clip_text("\n".join(files), MAX_CONTEXT_CHARS)
        note = "\n(listing truncated)" if clipped else ""
        return f"Workspace: {workspace}\nRepository files:\n{listing}{note}"

    # -------------------------------------------------------------- sanitising

    def _max_tasks(self) -> int:
        configured = getattr(self.client.settings, "max_parallel_tasks", HARD_MAX_TASKS)
        try:
            limit = int(configured)
        except (TypeError, ValueError):
            limit = HARD_MAX_TASKS
        return max(1, min(limit, HARD_MAX_TASKS))

    def _build_plan(self, data: dict[str, Any]) -> Plan:
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list):
            raw_tasks = []

        limit = self._max_tasks()
        tasks: list[TaskModel] = []
        used_ids: set[str] = set()

        for index, raw in enumerate(raw_tasks, start=1):
            if len(tasks) >= limit:
                break
            task = _clean_task(raw, index, used_ids)
            if task is not None:
                tasks.append(task)

        known = {task.id for task in tasks}
        for task in tasks:
            # Drop unknown ids and never let a task depend on itself.
            task.dependencies = [
                dep for dep in task.dependencies if dep in known and dep != task.id
            ]

        rationale = data.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = DEFAULT_RATIONALE

        return Plan(tasks=tasks, rationale=rationale.strip())

    # ---------------------------------------------------------------- sanitising


def _clean_task(
    raw: Any,
    index: int,
    used_ids: set[str],
) -> TaskModel | None:
    """Coerce one raw task mapping into a :class:`TaskModel`, or drop it."""
    if not isinstance(raw, dict):
        return None

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        return None

    raw_id = raw.get("id")
    raw_id = raw_id.strip() if isinstance(raw_id, str) else ""
    # Missing ids fall back to the 1-based position; duplicates get a
    # deterministic suffix so the returned plan always has unique ids.
    task_id = raw_id or str(index)
    if task_id in used_ids:
        suffix = 2
        candidate = f"{task_id}-{suffix}"
        while candidate in used_ids:
            suffix += 1
            candidate = f"{task_id}-{suffix}"
        task_id = candidate
    used_ids.add(task_id)

    files = _string_list(raw.get("files"))
    dependencies = [dep for dep in _string_list(raw.get("dependencies")) if dep != task_id]

    return TaskModel(
        id=task_id,
        description=description.strip(),
        files=files,
        dependencies=dependencies,
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
    return items


def _clip_text(text: str, limit: int) -> tuple[str, bool]:
    """Clip ``text`` to ``limit`` characters on a line boundary."""
    if len(text) <= limit:
        return text, False
    clipped = text[:limit]
    newline = clipped.rfind("\n")
    if newline > limit // 2:
        clipped = clipped[:newline]
    return clipped, True