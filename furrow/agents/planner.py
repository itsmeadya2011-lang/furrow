from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient
from furrow.state import StateManager

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state: StateManager | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or Settings()
        self.state = state or StateManager(settings=self.settings)

    def _get_workspace_context(self) -> str:
        files = self.client.list_files(self.settings.workspace)
        if not files:
            return "No files in workspace yet."
        # Limit to most relevant files
        relevant = [f for f in files if not f.startswith(".git") and not f.startswith("__pycache__")]
        relevant = sorted(relevant)[:50]
        return f"Existing files in workspace:\n" + "\n".join(f"  {f}" for f in relevant)

    def _get_history_context(self) -> str:
        history = self.state.history
        if not history:
            return "No previous cycles."
        recent = history[-3:]
        parts = ["Recent cycles:"]
        for entry in recent:
            cycle = entry["cycle"]
            plan = entry["plan"]
            test_result = entry["test_result"]
            task_count = len(plan.get("tasks", []))
            passed = test_result.get("passed", False)
            parts.append(f"  Cycle {cycle}: {task_count} tasks, tests {'passed' if passed else 'failed'}")
            if not passed:
                failures = test_result.get("failures", [])
                for f in failures[:3]:
                    parts.append(f"    - {f}")
        return "\n".join(parts)

    async def plan(self, goal: str) -> Plan:
        workspace_context = self._get_workspace_context()
        history_context = self._get_history_context()

        prompt = f"""{PLANNER_PROMPT}

{workspace_context}

{history_context}

Current goal: {goal}
Original goal: {self.state.original_goal or goal}

Return JSON only (no markdown, no explanation) with this shape:
{{
  "tasks": [
    {{
      "id": "1",
      "description": "Implement user authentication with JWT",
      "files": ["src/auth.py", "tests/test_auth.py"],
      "dependencies": []
    }}
  ],
  "rationale": "Brief explanation of the plan"
}}

If the goal is too large for one cycle, say so in rationale and break it into the most critical first slice.
"""
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
