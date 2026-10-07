from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.core.orchestrator import OrchestratorState


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()

    async def plan(self, goal: str, state: OrchestratorState | None = None) -> Plan:
        # Build context from previous cycles
        context = self._build_context(state) if state else ""

        prompt = f"{PLANNER_PROMPT}\n\n{context}\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    def _build_context(self, state: OrchestratorState) -> str:
        """Build context string from previous cycles."""
        if not state or not state.cycles:
            return ""

        parts = ["## Previous Cycles"]

        for cycle in state.cycles[-3:]:  # Last 3 cycles
            parts.append(f"\n### Cycle {cycle.cycle_num}")
            parts.append(f"Goal: {cycle.goal}")

            if cycle.plan:
                parts.append(f"Plan: {cycle.plan.rationale}")
                for task in cycle.plan.tasks:
                    status = state.all_tasks.get(task.id, task).status
                    parts.append(f"  - {task.id}: {task.description} [{status}]")

            if cycle.test_result:
                test_status = "PASSED" if cycle.test_result.passed else "FAILED"
                parts.append(f"Tests: {test_status} - {cycle.test_result.summary}")
                if cycle.test_result.failures:
                    for f in cycle.test_result.failures:
                        parts.append(f"  - Failure: {f}")

        # Add completed tasks summary
        completed = [t for t in state.all_tasks.values() if t.status == "completed"]
        if completed:
            parts.append("\n## Completed Tasks")
            for task in completed[-10:]:  # Last 10 completed
                parts.append(f"  - {task.id}: {task.description}")

        return "\n".join(parts)