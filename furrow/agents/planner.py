from __future__ import annotations

from typing import Any

from furrow.agents.parsing import parse_plan
from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.config import settings as global_settings
from furrow.llm import LLMClient


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.settings = settings if settings is not None else global_settings
        self.client = client if client is not None else LLMClient(settings=self.settings)
        self._last_plan_signature: str = ""
        self._consecutive_failures: int = 0

    async def plan(
        self, goal: str, context: str = "", task_history: list[dict[str, Any]] | None = None
    ) -> Plan:
        # Build enhanced context with task history
        enhanced_context = self._build_context(context, task_history)

        prompt = PLANNER_PROMPT + "\n\nGoal: " + goal
        if enhanced_context:
            prompt += "\n\nContext:\n" + enhanced_context

        model = self.settings.planner_model

        # Try up to 3 times with increasing strictness
        last_error = None
        for attempt in range(3):
            try:
                response = await self.client.complete(
                    prompt, system=PLANNER_SYSTEM, model=model
                )
                plan = parse_plan(response)

                # Check for no-progress (same plan signature)
                sig = self._plan_signature(plan)
                if sig == self._last_plan_signature:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 2:
                        # Add a hint to try a different approach
                        prompt += (
                            "\n\nIMPORTANT: The previous plan was identical. "
                            "Try a different approach or break down the tasks differently."
                        )
                else:
                    self._consecutive_failures = 0
                    self._last_plan_signature = sig

                return plan
            except ValueError as e:
                last_error = e
                # Increase strictness on each retry
                if attempt == 0:
                    prompt += (
                        "\n\nYour previous reply could not be parsed. "
                        "Reply with ONLY a JSON object matching the required shape, "
                        "no prose, no markdown fences."
                    )
                elif attempt == 1:
                    prompt += (
                        "\n\nCRITICAL: You must output valid JSON only. "
                        "No explanations, no markdown. Example format:\n{\n"
                        '  "tasks": [\n'
                        '    {"id": "1", "description": "Task description", '
                        '"files": ["file.py"], "dependencies": []}\n'
                        "  ],\n"
                        '  "rationale": "Brief explanation"\n}'
                    )

        raw = str(last_error) if last_error else "unknown error"
        raise ValueError(
            "Planner output could not be parsed after 3 attempts. "
            f"Last error: {raw[:500]}"
        )

    def _build_context(self, context: str, task_history: list[dict[str, Any]] | None) -> str:
        """Build enhanced context including task history."""
        parts = []
        if context:
            parts.append(context)
        if task_history:
            completed = [t for t in task_history if t.get("status") == "completed"]
            failed = [t for t in task_history if t.get("status") == "failed"]
            if completed:
                parts.append(
                    "Completed tasks:\n"
                    + "\n".join(f"  - {t['id']}: {t['description']}" for t in completed)
                )
            if failed:
                parts.append(
                    "Failed tasks:\n"
                    + "\n".join(
                        f"  - {t['id']}: {t['description']} (error: {t.get('error', 'unknown')})"
                        for t in failed
                    )
                )
        return "\n\n".join(parts) if parts else ""

    def _plan_signature(self, plan: Plan) -> str:
        """Generate a signature for a plan to detect duplicates."""
        return "|".join(f"{t.id}:{t.description}:{','.join(t.files)}" for t in plan.tasks)
