"""Planner agent: turn a goal into a normalised :class:`~furrow.config.Plan`.

The planner is the only place where untrusted model output becomes scheduling
state, so the plan is parsed defensively and then normalised:

* fenced/prose-wrapped JSON is recovered by :func:`furrow.parsing.parse_model`;
* one corrective retry is allowed before giving up;
* blank tasks are dropped, ids are renumbered, dangling dependencies are removed
  and dependency cycles are broken, so the scheduler can never be handed a
  graph it cannot execute.

An empty task list is a valid, meaningful result: it is how the run loop
terminates. It is never treated as an error.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, TaskModel
from furrow.config import settings as config_settings
from furrow.errors import StructuredOutputError
from furrow.llm import LLMClient
from furrow.parsing import parse_model

if TYPE_CHECKING:
    from furrow.config import Settings

_LOGGER = logging.getLogger(__name__)

_WHITE, _GREY, _BLACK = 0, 1, 2


def _correction_message(error: StructuredOutputError) -> str:
    """Build the retry instruction for a rejected planner response.

    Only ``error.message`` is echoed: it states precisely what was wrong
    (extraction failure or the pydantic validation summary) without pasting the
    model's raw output back into the conversation.
    """
    return (
        f"Your previous reply was rejected: {error.message}\n"
        "Reply with ONLY a raw JSON object matching the exact schema above: "
        'start with {"tasks": [ and end with } . Include a "rationale" string. '
        "Do not add prose, explanations, or markdown fences, and make sure "
        "every object and array is properly closed and double-quoted."
    )


def _find_back_edge(graph: dict[str, list[str]]) -> tuple[str, str] | None:
    """Return one ``(source, target)`` edge that closes a cycle, or ``None``.

    Iterative depth-first search, so a long dependency chain cannot exhaust the
    recursion limit. An edge pointing at a node still on the DFS stack is a back
    edge, i.e. an edge whose target can reach its source; dropping it cannot
    disconnect anything that is not part of that cycle.
    """
    colour = dict.fromkeys(graph, _WHITE)
    for root in graph:
        if colour[root] != _WHITE:
            continue
        colour[root] = _GREY
        # The second slot holds a one-shot iterator over the node's remaining
        # children, not an arbitrary object: the DFS consumes it and resumes it
        # on the next visit, so the annotation is what lets ``for child in
        # children`` type-check honestly.
        stack: list[tuple[str, Iterator[str]]] = [(root, iter(graph[root]))]
        while stack:
            node, children = stack[-1]
            descended = False
            for child in children:
                if child not in colour:
                    continue
                state = colour[child]
                if state == _GREY:
                    return node, child
                if state == _WHITE:
                    colour[child] = _GREY
                    stack.append((child, iter(graph[child])))
                    descended = True
                    break
            if not descended:
                colour[node] = _BLACK
                stack.pop()
    return None


def _break_cycles(graph: dict[str, list[str]]) -> tuple[dict[str, list[str]], list[str]]:
    """Drop back edges until the dependency graph is acyclic.

    Returns the repaired graph and a human-readable list of the edges removed,
    which the caller logs. Edges are visited in plan order so the result is
    deterministic for a given plan.
    """
    repaired = {node: list(deps) for node, deps in graph.items()}
    dropped: list[str] = []
    while True:
        edge = _find_back_edge(repaired)
        if edge is None:
            return repaired, dropped
        source, target = edge
        repaired[source].remove(target)
        dropped.append(f"{source} -> {target}")


class PlannerAgent:
    """Turns a natural-language goal into a normalised, schedulable plan."""

    def __init__(
        self,
        client: LLMClient,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings if settings is not None else config_settings

    @property
    def client(self) -> LLMClient:
        """The LLM client this planner calls."""
        return self._client

    @property
    def settings(self) -> Settings:
        """The settings this planner resolves models and limits from."""
        return self._settings

    async def plan(self, goal: str, *, max_tasks: int | None = None) -> Plan:
        """Produce a normalised plan for ``goal``.

        Raises :class:`~furrow.errors.StructuredOutputError` when the model does
        not return a parsable plan, even after one corrective retry.
        """
        prompt = PLANNER_PROMPT.format(goal=goal)
        model = self._settings.model_for("planner")

        response = await self._client.complete_detailed(
            prompt,
            system=PLANNER_SYSTEM,
            model=model,
        )
        if response.truncated:
            _LOGGER.warning(
                "planner response hit the token limit; the plan may be incomplete"
            )

        try:
            plan = parse_model(response.text, Plan)
        except StructuredOutputError as first_error:
            _LOGGER.warning("planner output rejected (%s); retrying once", first_error.message)
            retry = await self._client.complete_detailed(
                f"{prompt}\n\n{_correction_message(first_error)}",
                system=PLANNER_SYSTEM,
                model=model,
            )
            try:
                plan = parse_model(retry.text, Plan)
            except StructuredOutputError as second_error:
                # Forward ``raw`` so a caller logging ``err.raw`` sees the
                # offending response, not an empty string.
                raise StructuredOutputError(
                    "planner did not return a valid plan after one retry: "
                    f"{second_error.message}",
                    raw=second_error.raw,
                ) from second_error

        return self._normalise(plan, goal, max_tasks)

    def _normalise(self, plan: Plan, goal: str, max_tasks: int | None) -> Plan:
        """Make a model-produced plan safe for the scheduler.

        Steps, in order:

        1. drop tasks with a blank description;
        2. renumber surviving tasks to ``"1"``, ``"2"``, ... and remap every
           dependency through the old->new mapping, dropping references to ids
           that do not exist (duplicated ids resolve to the first task that
           claimed them, which keeps the mapping deterministic);
        3. drop the dependency edges that close a cycle, including
           self-dependencies;
        4. truncate to ``max_tasks`` (default ``settings.max_parallel_tasks``)
           and drop dependencies pointing at truncated tasks.
        """
        limit = max_tasks if max_tasks is not None else self._settings.max_parallel_tasks
        limit = max(limit, 0)

        kept = [task for task in plan.tasks if task.description and task.description.strip()]
        dropped_blank = len(plan.tasks) - len(kept)
        if dropped_blank:
            _LOGGER.info("dropped %d task(s) with an empty description", dropped_blank)

        # Renumber. First task to claim an old id owns that id's mapping, so a
        # duplicated id can never make a dependency ambiguous or self-referential.
        remap: dict[str, str] = {}
        for index, task in enumerate(kept, start=1):
            remap.setdefault(task.id, str(index))
        known_ids = set(remap.values())

        graph: dict[str, list[str]] = {}
        dangling: list[str] = []
        for index, task in enumerate(kept, start=1):
            task_id = str(index)
            resolved: list[str] = []
            for dependency in task.dependencies:
                target = remap.get(dependency)
                if target is None or target not in known_ids:
                    dangling.append(f"{task_id} -> {dependency}")
                    continue
                if target not in resolved:
                    resolved.append(target)
            graph[task_id] = resolved
        if dangling:
            _LOGGER.info("dropped unknown dependencies: %s", ", ".join(dangling))

        graph, cycle_edges = _break_cycles(graph)
        if cycle_edges:
            _LOGGER.warning(
                "planner produced a dependency cycle; dropped edges: %s",
                ", ".join(cycle_edges),
            )

        tasks = kept[:limit]
        if len(kept) > limit:
            _LOGGER.info(
                "truncated plan from %d to %d task(s) (max_tasks=%d)",
                len(kept),
                limit,
                limit,
            )
        # A dependency on a truncated task has nothing to point at any more.
        surviving_ids = {str(index) for index in range(1, len(tasks) + 1)}

        normalised: list[TaskModel] = []
        for index, task in enumerate(tasks, start=1):
            task_id = str(index)
            dependencies = [
                dependency
                for dependency in graph[task_id]
                if dependency in surviving_ids
            ]
            normalised.append(
                task.model_copy(update={"id": task_id, "dependencies": dependencies})
            )

        return Plan(tasks=normalised, rationale=self._rationale(plan.rationale, normalised, goal))

    @staticmethod
    def _rationale(rationale: str | None, tasks: list[TaskModel], goal: str) -> str:
        """Keep the model's rationale, synthesising one when it is blank."""
        if rationale and rationale.strip():
            return rationale.strip()
        if not tasks:
            return (
                "The planner returned no tasks, which is the loop's termination "
                f"signal: there is nothing left to do for the goal {goal!r}."
            )
        return (
            f"The planner returned {len(tasks)} task(s) for {goal!r} without a "
            "rationale; the plan was normalised by Furrow."
        )
