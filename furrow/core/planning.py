"""Planning-related functionality for the orchestrator."""

from __future__ import annotations

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.core.events import Event, EventBus, EventType
from furrow.llm import LLMClient


async def create_plan(
    goal: str,
    context: str,
    client: LLMClient,
    settings: Settings,
    event_bus: EventBus,
) -> Plan:
    """Create a plan for the given goal and context."""
    planner = PlannerAgent(client=client, settings=settings)
    try:
        plan = await planner.plan(goal, context=context)
    except Exception as e:
        event_bus.emit(Event(type=EventType.ERROR, message=str(e)))
        raise
    event_bus.emit(
        Event(
            type=EventType.PLAN_CREATED,
            data={"rationale": plan.rationale, "tasks": len(plan.tasks)},
        )
    )
    return plan


def merge_plan(plan: Plan, tasks: list[TaskModel], settings: Settings) -> set[str]:
    """Merge a new plan into existing tasks, returning IDs of tasks to skip."""
    skip_ids: set[str] = set()
    by_id = {t.id: t for t in tasks}
    max_retries = max(0, settings.max_task_retries)

    for index, pt in enumerate(plan.tasks):
        old = by_id.get(pt.id)
        if old is not None:
            old_desc = old.description
            old_files = list(old.files)
            # Preserve terminal states; only retry-eligible failed tasks may
            # be reset to pending. Blocked tasks must remain blocked so that
            # retry budgets and dependency outcomes are not lost across cycles.
            if old.status == "completed":
                if old_desc == pt.description and old_files == pt.files:
                    pt.status = "completed"
                    skip_ids.add(pt.id)
                else:
                    pt.status = "pending"
            elif old.status == "failed":
                if old.attempts <= max_retries:
                    pt.status = "pending"
                else:
                    pt.status = "failed"
            elif old.status == "blocked":
                pt.status = "blocked"
            else:
                pt.status = "pending"
            old.description = pt.description
            old.files = pt.files
            old.dependencies = pt.dependencies
            old.status = pt.status
            # A freshly planned task carries result=None and error=None,
            # so never let that wipe a result or error already recorded by
            # a previous cycle. Only adopt non-None values from the plan.
            if pt.result is not None:
                old.result = pt.result
            if pt.error is not None:
                old.error = pt.error
            plan.tasks[index] = old
        else:
            tasks.append(pt)

    return skip_ids


def plan_waves(tasks: list[TaskModel]) -> list[list[TaskModel]]:
    """Plan execution waves based on task dependencies."""
    ids = {t.id for t in tasks}
    waves: list[list[TaskModel]] = []
    done: set[str] = set()
    remaining = list(tasks)
    while remaining:
        wave = [
            t
            for t in remaining
            if all(d in done or d not in ids for d in t.dependencies)
        ]
        if not wave:
            for t in remaining:
                t.status = "blocked"
            break
        waves.append(wave)
        for t in wave:
            done.add(t.id)
        remaining = [t for t in remaining if t not in wave]
    return waves