"""Turn a goal into a validated :class:`~furrow.config.Plan`.

The planner is the boundary between free-form model output and the orchestrator, so it
owns two responsibilities: recovering a JSON object from whatever the model replied, and
sanitising that object into a plan the orchestrator can schedule without having to defend
against duplicate ids or dangling dependency edges.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient, extract_json

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger("furrow.agents.planner")

#: Upper bound on the number of workspace paths embedded in a planner prompt. The goal
#: still reaches the model, just without an unbounded listing pushed in front of it.
MAX_LISTED_FILES = 200

#: Extra attempts made when a reply carries no usable JSON object. One retry recovers
#: from a stray code fence or a truncated completion without looping.
PLAN_PARSE_RETRIES = 1

#: Characters of the offending reply quoted in the error message.
PREVIEW_CHARS = 200

#: Appended to the retry prompt: the cheapest way to fix a fenced or chatty reply.
JSON_ONLY_INSTRUCTION = (
    "Your previous reply could not be parsed. Reply with a single valid JSON object and "
    "nothing else: no prose, no explanation, no markdown code fences."
)

#: Declared here rather than in ``prompts.py`` so the dependency contract lives next to
#: the sanitiser that enforces it. The orchestrator schedules by dependency waves and
#: runs tasks without an edge between them concurrently, which this must state.
DEPENDENCY_RULES = """
Dependency rules:
- "dependencies" lists the ids of the tasks in this plan that must complete first.
- Tasks that do not depend on each other run concurrently, so two tasks touching the
  same files must declare a dependency on one another.
- Every id in "dependencies" must also appear as a task "id" in the same plan.
"""


class PlannerAgent:
    """Plan a goal into a set of tasks the orchestrator can execute directly."""

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        # The fallback is spelled out at the call site instead of being left implicit in
        # LLMClient, so the no-argument construction path stays obvious.
        self.client = client if client is not None else LLMClient(settings=settings or default_settings)
        # The client's own resolved settings win (LLMClient substitutes its default for
        # None); keeping them here makes model selection inspectable and overridable.
        self.settings: Settings = getattr(self.client, "settings", None) or settings or default_settings

    async def plan(self, goal: str) -> Plan:
        """Return a sanitised plan for ``goal``.

        Raises:
            ValueError: If no reply (including the retry) yields a JSON object that can
                be turned into a plan. The message names the goal, the reason and a
                truncated preview of the reply.
        """
        prompt = self._build_prompt(goal)
        response = await self.client.complete(prompt, model=self.settings.planner_model)
        try:
            return self._to_plan(response, goal)
        except ValueError as exc:
            # Kept under a name other than the ``as`` target: Python unbinds the latter
            # once the block ends, and the retry wants to chain onto this failure.
            first_error = exc
            logger.warning("Planner reply for goal %r is unusable: %s", goal, first_error)

        last_error: ValueError | None = None
        for attempt in range(1, PLAN_PARSE_RETRIES + 1):
            retry_prompt = f"{prompt}\n\n{JSON_ONLY_INSTRUCTION}"
            retry_response = await self.client.complete(
                retry_prompt, model=self.settings.planner_model
            )
            try:
                plan = self._to_plan(retry_response, goal)
            except ValueError as exc:
                last_error = exc
                logger.warning("Planner retry %d for goal %r is unusable: %s", attempt, goal, exc)
                continue
            logger.warning("Planner retry %d recovered a usable plan for goal %r.", attempt, goal)
            return plan

        if last_error is not None:
            raise last_error from first_error
        raise first_error

    # ---------------------------------------------------------------------- prompt --

    def _build_prompt(self, goal: str) -> str:
        """Assemble the planner prompt, keeping the goal last where attention is strongest."""
        return f"{PLANNER_PROMPT}{DEPENDENCY_RULES}\n\n{self._workspace_listing()}\n\nGoal: {goal}\n"

    def _workspace_listing(self) -> str:
        """Return a bounded listing of workspace paths for the model to plan against.

        Only paths are listed, never contents: the model needs to know what exists, and
        reading every file would dwarf the prompt.
        """
        workspace = self.settings.workspace
        try:
            files = self.client.list_files(workspace)
        except OSError as exc:  # unreadable workspace: plan from the goal alone
            logger.warning("Could not list workspace %s: %s", workspace, exc)
            return f"Workspace listing unavailable for {workspace}: {exc}"
        if not files:
            return f"Workspace {workspace} contains no files."
        shown = files[:MAX_LISTED_FILES]
        hidden = len(files) - len(shown)
        note = f", {hidden} more not shown" if hidden > 0 else ""
        header = f"Workspace files relative to {workspace} ({len(files)} total{note}):"
        return "\n".join([header, ", ".join(shown)])

    # --------------------------------------------------------------------- parsing --

    def _to_plan(self, response: str, goal: str) -> Plan:
        """Parse one model reply into a sanitised :class:`Plan`.

        Raises:
            ValueError: If the reply carries no JSON object, or carries something that is
                not a usable plan.
        """
        try:
            data = extract_json(response)
        except ValueError as exc:
            raise ValueError(self._failure(goal, str(exc), response)) from exc
        if not isinstance(data, Mapping):
            reason = f"expected a JSON object, got {type(data).__name__}"
            # The ValueError is deliberate: a malformed *model reply* is a value problem
            # for the caller (the orchestrator retries the cycle), not a programming error.
            raise ValueError(self._failure(goal, reason, response))  # noqa: TRY004
        raw_tasks = data.get("tasks")
        if not isinstance(raw_tasks, list):
            reason = f"'tasks' must be a list, got {type(raw_tasks).__name__}"
            raise ValueError(self._failure(goal, reason, response))  # noqa: TRY004

        tasks = self._sanitize(raw_tasks)
        if not tasks:
            # An empty plan is a legitimate answer ("goal already done"), but it is also
            # what a heavily malformed reply degrades to, so make the difference visible.
            logger.warning("Plan for goal %r contains no usable tasks.", goal)
        rationale = str(data.get("rationale") or "").strip()
        return Plan(tasks=tasks, rationale=rationale)

    @staticmethod
    def _failure(goal: str, reason: str, response: Any) -> str:
        """Return the operator-facing message for an unusable plan reply."""
        preview = " ".join(str(response).split())
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS] + "..."
        return f"Failed to plan goal {goal!r}: {reason} | response preview: {preview!r}"

    # ---------------------------------------------------------------- sanitisation --

    def _sanitize(self, raw_tasks: list[Any]) -> list[TaskModel]:
        """Drop unusable tasks and dependency edges, logging everything removed."""
        kept: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, entry in enumerate(raw_tasks):
            if not isinstance(entry, Mapping):
                logger.warning("Task #%d dropped: not a JSON object (%r)", index + 1, entry)
                continue
            description = str(entry.get("description") or "").strip()
            if not description:
                logger.warning("Task #%d dropped: empty description", index + 1)
                continue
            task_id = str(entry.get("id") or "").strip() or self._synthesise_id(index, seen)
            if task_id in seen:
                logger.warning("Task %r dropped: duplicate id", task_id)
                continue
            seen.add(task_id)
            kept.append(
                {
                    "id": task_id,
                    "description": description,
                    "files": self._strings(entry.get("files")),
                    "dependencies": [
                        str(dep).strip() for dep in self._sequence(entry.get("dependencies"))
                    ],
                }
            )

        # Only ids that survived the filters are valid targets, and a dependency may point
        # forward, so the edges are resolved once every id is known.
        valid_ids = {task["id"] for task in kept}
        tasks: list[TaskModel] = []
        for task in kept:
            dependencies: list[str] = []
            for dep in task["dependencies"]:
                if not dep:
                    logger.warning("Task %r: blank dependency dropped.", task["id"])
                elif dep == task["id"]:
                    logger.warning("Task %r: self-dependency dropped.", task["id"])
                elif dep not in valid_ids:
                    logger.warning(
                        "Task %r: dependency %r is not a task in this plan; dropped.",
                        task["id"],
                        dep,
                    )
                elif dep not in dependencies:
                    dependencies.append(dep)
            task["dependencies"] = dependencies
            tasks.append(TaskModel(**task))
        return tasks

    @staticmethod
    def _synthesise_id(index: int, seen: set[str]) -> str:
        """Return a stable id for a task the model left without one."""
        base = str(index + 1)
        candidate = base
        suffix = 2
        while candidate in seen:
            candidate = f"{base}-{suffix}"
            suffix += 1
        logger.warning("Task #%d had no id; using %r.", index + 1, candidate)
        return candidate

    @staticmethod
    def _sequence(value: Any) -> list[Any]:
        """Coerce a model supplied value into a list, tolerating a bare scalar."""
        if isinstance(value, (str, bytes)) or value is None:
            return [] if value is None else [value]
        if isinstance(value, Sequence):
            return list(value)
        return []

    @classmethod
    def _strings(cls, value: Any) -> list[str]:
        """Coerce a model supplied value into a list of non-blank strings."""
        return [text for text in (str(item).strip() for item in cls._sequence(value)) if text]