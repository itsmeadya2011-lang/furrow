from __future__ import annotations

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings, TaskModel
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(
        self,
        goal: str,
        context: str = "",
        history: list | None = None,
        tasks: list[TaskModel] | None = None,
    ) -> Plan:
        """Generate a plan for the given goal.

        Args:
            goal: The high-level goal to achieve.
            context: Immediate context (e.g., test failures from the last cycle).
            history: List of HistoryEntry objects from previous cycles.
            tasks: Current list of tasks with their statuses.
        """
        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal

        # Build rich context from history and current task state
        context_parts = []
        if context:
            context_parts.append(f"Immediate context (e.g., test failures):\n{context}")

        if history:
            context_parts.append(self._format_history(history))

        if tasks:
            context_parts.append(self._format_task_state(tasks))

        if context_parts:
            prompt += "\n\nContext:\n" + "\n\n---\n\n".join(context_parts)

        model = self.settings.planner_model
        response = await self.client.complete(prompt, system=PLANNER_SYSTEM, model=model)
        try:
            return parse_plan(response)
        except ValueError as first:
            stricter = (
                prompt
                + "\n\nYour previous reply could not be parsed. Reply with ONLY a JSON object "
                + "matching the required shape, no prose, no markdown fences."
            )
            response2 = await self.client.complete(stricter, system=PLANNER_SYSTEM, model=model)
            try:
                return parse_plan(response2)
            except ValueError as second:
                raw = response2 if isinstance(response2, str) else str(response2)
                raise ValueError(
                    "Planner output could not be parsed. "
                    f"attempt 1: {first}. attempt 2: {second}. Response: {raw[:500]}"
                ) from second

    def _format_history(self, history: list) -> str:
        """Format recent history entries for the planner prompt."""
        if not history:
            return ""
        # Show last 5 cycles to avoid overwhelming the context
        recent = history[-5:]
        lines = ["Recent cycle history:"]
        for entry in recent:
            lines.append(
                f"  Cycle {entry.cycle}: {entry.plan_rationale[:120]} "
                f"(tests_passed={entry.tests_passed}, tasks={len(entry.task_summaries)})"
            )
            if entry.test_summary:
                lines.append(f"    Test summary: {entry.test_summary[:200]}")
        return "\n".join(lines)

    def _format_task_state(self, tasks: list[TaskModel]) -> str:
        """Format current task state for the planner prompt."""
        if not tasks:
            return ""
        lines = ["Current task state:"]
        for task in tasks:
            status = task.status
            files = f" [{', '.join(task.files)}]" if task.files else ""
            deps = f" (deps: {', '.join(task.dependencies)})" if task.dependencies else ""
            result = f" -> {task.result[:100]}" if task.result else ""
            error = f" [ERROR: {task.error[:100]}]" if task.error else ""
            lines.append(
                f"  {task.id}: {status}{files}{deps}{result}{error} - {task.description[:120]}"
            )
        return "\n".join(lines)
