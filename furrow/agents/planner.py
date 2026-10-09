from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient, _strip_json_markers

if TYPE_CHECKING:
    from furrow.config import Settings

log = structlog.get_logger(__name__)


class PlannerAgent:
    """Breaks a high-level goal into 1-5 parallelizable, independent tasks.

    The planner receives project context (file listing and recent task results)
    so it can produce informed, non-conflicting plans.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, prior_tasks: list[TaskModel] | None = None) -> Plan:
        project_context = await self._build_project_context(prior_tasks)

        prompt = (
            f"{PLANNER_PROMPT}\n\n"
            f"{project_context}\n"
            f"Goal: {goal}\n"
        )

        response = await self.client.complete(
            prompt,
            model=self.client.settings.planner_model,
        )

        stripped = _strip_json_markers(response)
        try:
            data = json.loads(stripped)
            plan = Plan(**data)
            log.info("Plan created", tasks=len(plan.tasks), rationale=plan.rationale)
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Failed to parse plan from LLM: {e}\nResponse: {response}"
            )

    async def _build_project_context(self, prior_tasks: list[TaskModel] | None) -> str:
        """Build a context string with file listing and prior cycle results."""
        lines: list[str] = ["## Project Context\n"]

        # List relevant files (exclude noise)
        files = self.client.list_files(self.client.settings.workspace)
        ignore_prefixes = (
            ".git",
            "__pycache__",
            ".venv",
            "node_modules",
            ".furrow-state",
        )
        relevant = [f for f in files if not f.startswith(ignore_prefixes)]

        if relevant:
            lines.append("### Existing project files:")
            for f in sorted(relevant)[:50]:  # cap to avoid huge prompts
                lines.append(f"  - {f}")
            if len(relevant) > 50:
                lines.append(f"  ... and {len(relevant) - 50} more files")
        else:
            lines.append("### No existing project files found.")

        # Include prior task results for continuity
        if prior_tasks:
            lines.append("")
            lines.append("### Prior cycle tasks:")
            for t in prior_tasks:
                lines.append(f"  - [{t.status}] {t.description}")

        return "\n".join(lines)
