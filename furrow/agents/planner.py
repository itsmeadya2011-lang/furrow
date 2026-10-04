"""Planner agent: turn a natural-language goal into a validated, normalised batch of tasks.

The planner is the only component that decides *what* furrow does, so this module carries
two responsibilities:

1. Talk to the model (role prompt as ``system``, goal plus a short workspace listing as the
   user message) and parse the reply with :func:`furrow.jsonutil.parse_model_json`, which
   copes with markdown fences and surrounding prose. Raw ``json.loads`` used to run here and
   every fenced response crashed the whole run with a ``JSONDecodeError`` the orchestrator
   has no handler for.
2. Make the parsed plan internally consistent via :func:`_normalise`, because models emit
   blank ids, duplicate ids, dependency edges pointing at tasks that do not exist, and
   eleven tasks when the prompt asked for at most five. All of those are repaired in one
   place instead of being discovered halfway through execution.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import get_settings
from furrow.jsonutil import extract_json, parse_model_json, truncate
from furrow.llm import LLMClient
from furrow.models import Plan

if TYPE_CHECKING:
    from furrow.config import Settings

__all__ = ["MAX_TASKS_PER_PLAN", "PlannerAgent"]

#: Upper bound on tasks in a single plan. The prompt promises 1-5; without a code-side cap
#: a verbose model produced 11 tasks, which the orchestrator then ran as one wave and whose
#: "ninth task" could never be traced back to anything.
MAX_TASKS_PER_PLAN = 5

#: Cap on the Python files listed to the model as workspace context. Enough to ground a
#: plan in the real layout without spending the whole completion budget on a file listing.
MAX_CONTEXT_FILES = 40


def _normalise(plan: Plan) -> Plan:
    """Make a parsed :class:`~furrow.models.Plan` internally consistent.

    Truncates the task list to :data:`MAX_TASKS_PER_PLAN` (noting the truncation in
    ``rationale``), gives every task a unique non-blank id, and drops dependency entries
    that name a task id which is not in the plan. A dangling dependency would otherwise
    never be satisfied and would stall the task forever.

    Raises:
        ValueError: if the plan contains no tasks. An empty plan used to be read by the
            orchestrator as "goal complete", so a model that answered with an empty list
            silently ended the run; it is now a loud, actionable failure.
    """
    if not plan.tasks:
        raise ValueError(
            "the planner returned no tasks for this goal, so there is nothing to execute; "
            "re-run with a more concrete goal or raise FURROW_MAX_TOKENS if the reply was "
            "cut short"
        )

    total = len(plan.tasks)
    tasks = plan.tasks[:MAX_TASKS_PER_PLAN]
    if total > MAX_TASKS_PER_PLAN:
        plan.rationale = (
            f"{plan.rationale} [furrow truncated this plan to the first {MAX_TASKS_PER_PLAN} "
            f"of {total} tasks; the remaining {total - MAX_TASKS_PER_PLAN} were dropped]."
        ).strip()

    taken: set[str] = set()
    for index, task in enumerate(tasks, start=1):
        ident = task.id.strip()
        if not ident:
            ident = f"task-{index}"
        if ident in taken:
            # Duplicate ids make dependency edges ambiguous, so suffix until unique.
            suffix = 2
            while f"{ident}-{suffix}" in taken:
                suffix += 1
            ident = f"{ident}-{suffix}"
        task.id = ident
        taken.add(ident)

    for task in tasks:
        task.dependencies = [dep for dep in task.dependencies if dep in taken]

    plan.tasks = tasks
    return plan


def _blank_missing_ids(payload: object) -> object:
    """Return `payload` with a blank ``id`` injected into every task dict that lacks one.

    ``TaskModel.id`` is a required field, so a reply that omits it fails pydantic validation
    before :func:`_normalise` ever runs -- and the model omitting an id is exactly the case
    normalisation is meant to absorb. A blank id validates, and normalisation then names the
    task ``task-N``.
    """
    if not isinstance(payload, dict):
        return payload
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return payload
    patched: list[object] = []
    for task in tasks:
        if isinstance(task, dict) and not str(task.get("id") or "").strip():
            task = {**task, "id": ""}
        patched.append(task)
    return {**payload, "tasks": patched}


class PlannerAgent:
    """Decompose a goal into an ordered batch of tasks.

    Settings are resolved eagerly: ``settings=None`` falls back to the process-wide
    :func:`furrow.config.get_settings` singleton rather than being stored, so a zero-argument
    ``PlannerAgent()`` works (it used to raise ``AttributeError`` on the first
    ``self.client.settings.planner_model``).
    """

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        """Build a planner. `client` is shared with the orchestrator when one is supplied."""
        self.settings: Settings = settings if settings is not None else get_settings()
        self.client = client or LLMClient(self.settings)

    async def plan(self, goal: str) -> Plan:
        """Ask the model to plan `goal` and return a normalised plan.

        Args:
            goal: The natural-language objective for this cycle.

        Returns:
            A :class:`~furrow.models.Plan` with at most :data:`MAX_TASKS_PER_PLAN` tasks,
            unique ids and no dangling dependencies.

        Raises:
            ValueError: if the reply is not a parseable ``Plan``, if it contains no tasks, or
                if the model response could not be used. The message embeds at most 500
                chars of the reply, not the whole multi-thousand-token body.
        """
        prompt = self._build_prompt(goal)
        response = await self.client.complete(
            prompt,
            system=PLANNER_PROMPT,
            model=self.settings.models()["planner"],
        )
        try:
            return _normalise(self._parse(response))
        except ValueError as exc:
            raise ValueError(
                f"planner returned output that is not a Plan: {exc}; "
                f"raw response: {truncate(response, 500)}"
            ) from exc

    def _parse(self, response: str) -> Plan:
        """Validate `response` as a :class:`~furrow.models.Plan`, tolerating missing task ids.

        The first attempt is the plain :func:`~furrow.jsonutil.parse_model_json` call, which
        handles markdown fences and surrounding prose. If it fails only because a task dict
        left out its ``id``, the reply is re-parsed through
        :func:`_blank_missing_ids` so normalisation can name the task instead of the whole
        cycle dying on a pydantic ``Field required`` error.
        """
        try:
            return parse_model_json(response, Plan)
        except ValueError:
            repaired = json.dumps(_blank_missing_ids(extract_json(response)))
            return parse_model_json(repaired, Plan)

    def _build_prompt(self, goal: str) -> str:
        """Compose the user message: the goal plus a short look at the workspace."""
        workspace = self.settings.workspace_path()
        try:
            listing = self.client.list_files(
                workspace, pattern="*.py", max_entries=MAX_CONTEXT_FILES
            )
        except OSError as exc:  # unreadable or racing tree; the plan does not depend on it
            listing = []
            files_block = f"<workspace listing unavailable: {exc}>"
        else:
            files_block = "\n".join(f"  {path}" for path in listing) or "  <no Python files>"
        return (
            f"Workspace root: {workspace}\n"
            f"Existing Python files (relative to that root):\n{files_block}\n\n"
            f"Goal:\n{goal}"
        )
