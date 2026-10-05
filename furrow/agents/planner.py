from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)

_MAX_TASKS = 5


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        response = await self.client.complete(
            prompt, system=PLANNER_SYSTEM_PROMPT, model=self.client.settings.planner_model
        )
        try:
            data = self.client.extract_json(response)
        except (TypeError, ValueError) as e:
            if isinstance(e, TypeError):
                # extract_json refuses a non-str reply with TypeError (its argument is
                # typed str), and there is then no response text to quote.  Callers are
                # written against ValueError, so still raise it.
                reply_type = type(response).__name__
                log.warning("planner_malformed_reply", reply_type=reply_type, error=str(e))
                message = f"Failed to parse plan from LLM: {e} (reply was {reply_type}, not a str)"
            else:
                snippet = response[:300] + ("..." if len(response) > 300 else "")
                message = f"Failed to parse plan from LLM: {e}\nResponse snippet: {snippet}"
            raise ValueError(message) from e

        tasks_data = data.get("tasks", [])
        if not isinstance(tasks_data, list):
            raise ValueError("Plan 'tasks' must be a list")  # noqa: TRY004

        # Normalize and validate tasks
        seen_ids: dict[str, int] = {}
        normalized_tasks: list[TaskModel] = []
        all_ids: set[str] = set()

        for idx, task_data in enumerate(tasks_data):
            if not isinstance(task_data, dict):
                continue
            task_id = str(task_data.get("id", "")).strip()
            if not task_id:
                task_id = f"task-{idx + 1}"

            # Handle duplicate IDs
            if task_id in seen_ids:
                seen_ids[task_id] += 1
                task_id = f"{task_id}-{seen_ids[task_id]}"
            else:
                seen_ids[task_id] = 0
            all_ids.add(task_id)

            description = str(task_data.get("description", "")).strip()
            if not description:
                description = f"implement {task_id}"

            files = task_data.get("files", [])
            if not isinstance(files, list):
                files = []
            files = [str(f).strip() for f in files if str(f).strip()]

            dependencies = task_data.get("dependencies", [])
            if not isinstance(dependencies, list):
                dependencies = []
            dependencies = [str(d).strip() for d in dependencies if str(d).strip()]

            # Drop self-dependencies
            dependencies = [d for d in dependencies if d != task_id]

            normalized_tasks.append(
                TaskModel(
                    id=task_id,
                    description=description,
                    files=files,
                    dependencies=dependencies,
                )
            )

        # Drop dependencies pointing to unknown IDs
        for task in normalized_tasks:
            task.dependencies = [d for d in task.dependencies if d in all_ids]

        # Enforce 1..5 task cap. Report any truncation so the loss is visible to
        # the loop's consumers (CLI plan panel, the websocket "plan" event and
        # /api/state all carry Plan.rationale), mirroring how the orchestrator's
        # bounded-digest helpers call out what they omit.
        total = len(normalized_tasks)
        dropped = total - _MAX_TASKS
        if dropped > 0:
            log.warning(
                "planner_truncated_tasks",
                dropped=dropped,
                kept=_MAX_TASKS,
                total=total,
            )
            normalized_tasks = normalized_tasks[:_MAX_TASKS]

        if not normalized_tasks:
            raise ValueError("Plan must contain at least one task")

        rationale = str(data.get("rationale", "")).strip()
        if not rationale:
            rationale = "planner"
        if dropped > 0:
            rationale = f"{rationale}; {dropped} task(s) dropped (kept {_MAX_TASKS} of {total})"

        plan = Plan(tasks=normalized_tasks, rationale=rationale)
        return plan
