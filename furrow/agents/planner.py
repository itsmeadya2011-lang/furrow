from __future__ import annotations

import json

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings, TaskModel
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)

    async def plan(self, goal: str, context: str = "", existing_tasks: list[TaskModel] | None = None) -> Plan:
        prompt = self._build_prompt(goal, context, existing_tasks)

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

    def _build_prompt(self, goal: str, context: str, existing_tasks: list[TaskModel] | None) -> str:
        """Build a detailed prompt for the planner with context about existing tasks."""
        parts = [PLANNER_PROMPT]

        # Add context about existing tasks if any
        if existing_tasks:
            parts.append("\n\nExisting tasks from previous cycles:")
            for task in existing_tasks:
                status = task.status
                files = f" (files: {', '.join(task.files)})" if task.files else ""
                deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
                result = f" - result: {task.result}" if task.result else ""
                error = f" - error: {task.error}" if task.error else ""
                parts.append(f"  - {task.id}: {task.description} [{status}]{files}{deps}{result}{error}")

        parts.append(f"\n\nGoal: {goal}")

        if context:
            parts.append(f"\n\nContext:\n{context}")

        parts.append("""
\nReturn JSON only (no markdown, no explanation) with this shape:
{
  "tasks": [
    {
      "id": "1",
      "description": "Implement user authentication with JWT",
      "files": ["src/auth.py", "tests/test_auth.py"],
      "dependencies": []
    }
  ],
  "rationale": "Brief explanation of the plan"
}

Guidelines:
- Break the goal into 1-5 parallelizable tasks
- Each task should be completable in 1-3 tool call rounds by a worker
- Be specific enough that a worker can implement it without ambiguity
- Use numeric IDs (1, 2, 3...) for new tasks
- Reference existing task IDs for dependencies
- If the goal is too large for one cycle, say so in rationale and break it into the most critical first slice
- Prefer modifying existing tasks over creating new ones when the work is similar
""")

        return "".join(parts)
