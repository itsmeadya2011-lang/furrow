from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, State, TaskStatus
from furrow.config import get_logger
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("planner")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, state: State | None = None) -> Plan:
        # Build context from existing state
        context = ""
        if state and state.tasks:
            completed = [t for t in state.tasks if t.status == TaskStatus.COMPLETED]
            failed = [t for t in state.tasks if t.status == TaskStatus.FAILED]
            pending = [t for t in state.tasks if t.status == TaskStatus.PENDING]
            in_progress = [t for t in state.tasks if t.status == TaskStatus.IN_PROGRESS]
            
            context = "\n\nCurrent State:"
            if completed:
                context += f"\nCompleted ({len(completed)}): " + ", ".join(f"{t.id}: {t.description[:50]}" for t in completed)
            if failed:
                context += f"\nFailed ({len(failed)}): " + ", ".join(f"{t.id}: {t.description[:50]}" for t in failed)
            if in_progress:
                context += f"\nIn Progress ({len(in_progress)}): " + ", ".join(f"{t.id}: {t.description[:50]}" for t in in_progress)
            if pending:
                context += f"\nPending ({len(pending)}): " + ", ".join(f"{t.id}: {t.description[:50]}" for t in pending)
            
            context += "\n\nOnly plan NEW tasks that are not already covered. Consider dependencies on completed tasks."

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_generated", tasks=len(plan.tasks), rationale=plan.rationale)
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
