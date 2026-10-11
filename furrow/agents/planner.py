from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import OrchestratorState, Plan, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings

    async def plan(self, goal: str, state: OrchestratorState | None = None) -> Plan:
        # Build context from state
        context_parts = [f"Goal: {goal}"]
        
        if state:
            if state.cycles > 0:
                context_parts.append(f"\nPrevious cycles: {state.cycles}")
            
            # Include completed tasks
            completed = [t for t in state.tasks if t.status == "completed"]
            if completed:
                context_parts.append("\nCompleted tasks:")
                for task in completed:
                    context_parts.append(f"  - {task.id}: {task.description}")
                    if task.result:
                        context_parts.append(f"    Result: {task.result[:200]}")
            
            # Include failed tasks
            failed = [t for t in state.tasks if t.status == "failed"]
            if failed:
                context_parts.append("\nFailed tasks (need retry/fix):")
                for task in failed:
                    context_parts.append(f"  - {task.id}: {task.description}")
                    if task.result:
                        context_parts.append(f"    Error: {task.result[:200]}")
            
            # Include last test result
            if state.last_test_result and not state.last_test_result.passed:
                context_parts.append("\nLast test failures:")
                for failure in state.last_test_result.failures:
                    context_parts.append(f"  - {failure}")

        prompt = f"{PLANNER_PROMPT}\n\n" + "\n".join(context_parts) + "\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
