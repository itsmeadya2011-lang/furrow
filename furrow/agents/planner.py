"""Planner agent: turns a goal into a small, validated :class:`~furrow.config.Plan`.

The planner is the least forgiving stage, because every task it emits becomes
real work elsewhere. Its reply is therefore validated rather than trusted: one
repair request is made when the JSON is malformed, the documented task-count
limit is enforced in code rather than left to the prompt, unknown dependency
ids are dropped so a dependency-aware scheduler cannot deadlock, and statuses
are normalised through :func:`~furrow.agents.parsing.coerce_status`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.agents.parsing import coerce_status, parse_json_response
from furrow.agents.prompts import MAX_TASKS, PLANNER_PROMPT, PLANNER_SYSTEM_PROMPT
from furrow.config import Plan, get_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

__all__ = ["PlannerAgent"]


class PlannerAgent:
    """Produces an ordered plan for a goal.

    Attributes:
        settings: Resolved configuration: the explicit ``settings`` argument
            when one is given, otherwise the client's, otherwise the
            process-global :func:`~furrow.config.get_settings` singleton.
        client: The LLM client used for planning and for the repair request.
    """

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        """Resolve the configuration and the client to plan with.

        Args:
            client: Client to reuse. Its ``settings`` are consulted only
                when no ``settings`` argument is supplied.
            settings: Configuration that takes precedence over the client's,
                and that builds the client when none is given.
        """
        self.settings = settings or (client.settings if client is not None else get_settings())
        self.client = client or LLMClient(settings=self.settings)

    async def plan(self, goal: str) -> Plan:
        """Plan ``goal``, repairing one malformed reply.

        Args:
            goal: The user's objective, in their own words.

        Returns:
            A :class:`~furrow.config.Plan` holding at most
            :data:`~furrow.agents.prompts.MAX_TASKS` tasks, each with a
            normalised status and only dependency ids that exist in the plan.

        Raises:
            ValueError: If the model returns unparsable JSON twice, or returns a
                plan with no tasks at all.
            furrow.llm.LLMError: If a completion request fails.
        """
        prompt = self.build_prompt(goal)
        response = await self.client.complete(
            prompt,
            system=PLANNER_SYSTEM_PROMPT,
            model=self.client.settings.planner_model,
        )
        try:
            plan = parse_json_response(response, Plan, "planner response")
        except ValueError as first_error:
            response = await self._repair(goal, first_error, response)
            try:
                plan = parse_json_response(response, Plan, "planner repair response")
            except ValueError as exc:
                raise ValueError(
                    f"planner returned no usable plan after a repair attempt: {exc}"
                ) from exc
        self._enforce_limits(plan)
        return plan

    def build_prompt(self, goal: str) -> str:
        """Assemble the planner prompt for ``goal``.

        Args:
            goal: The user's objective.

        Returns:
            The prompt text: the plan template followed by the goal.
        """
        return f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"

    async def _repair(self, goal: str, error: ValueError, offending: str) -> str:
        """Ask the model once to re-emit a corrected JSON plan.

        Args:
            goal: The original objective, repeated so the model does not have to
                reconstruct it.
            error: Why the first reply was rejected.
            offending: The reply that failed to parse.

        Returns:
            The corrected reply, which is still subject to validation.
        """
        prompt = "\n".join(
            [
                PLANNER_PROMPT,
                "",
                f"Goal: {goal}",
                "",
                "Your previous reply was rejected and must be rewritten.",
                f"Reason: {error}",
                "Previous reply:",
                offending,
                "",
                f"Reply again with a single corrected JSON object and nothing "
                f"else, containing between 1 and {MAX_TASKS} tasks.",
            ]
        )
        return await self.client.complete(
            prompt,
            system=PLANNER_SYSTEM_PROMPT,
            model=self.client.settings.planner_model,
        )

    def _enforce_limits(self, plan: Plan) -> None:
        """Clamp, clean up and annotate a validated plan in place.

        Truncates to :data:`~furrow.agents.prompts.MAX_TASKS` tasks, refuses an
        empty plan, normalises each status, and drops dependency ids that name
        a missing task or the task itself, since either would deadlock a
        dependency-aware scheduler. A note is appended to ``rationale`` whenever
        something was changed, so the truncation is never silent.

        Args:
            plan: The validated plan to normalise.

        Raises:
            ValueError: If ``plan`` contains no tasks.
        """
        notes: list[str] = []
        if len(plan.tasks) > MAX_TASKS:
            dropped = len(plan.tasks) - MAX_TASKS
            plan.tasks = plan.tasks[:MAX_TASKS]
            notes.append(f"Truncated {dropped} task(s) to respect the {MAX_TASKS}-task limit.")
        if not plan.tasks:
            raise ValueError("planner returned no tasks for the goal")

        known = {task.id for task in plan.tasks}
        removed_deps = 0
        for task in plan.tasks:
            task.status = coerce_status(task.status)
            kept = [dep for dep in task.dependencies if dep in known and dep != task.id]
            removed_deps += len(task.dependencies) - len(kept)
            task.dependencies = kept
        if removed_deps:
            notes.append(
                f"Dropped {removed_deps} unknown or self-referential dependency "
                f"id(s) to keep the plan schedulable."
            )
        if notes:
            plan.rationale = f"{plan.rationale.rstrip()} {' '.join(notes)}".strip()
