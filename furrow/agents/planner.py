from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.logger = get_logger("planner")

    async def plan(self, goal: str, context: str = "") -> Plan:
        # Get workspace file context
        workspace_context = await self._get_workspace_context()
        
        prompt = f"{PLANNER_PROMPT}\n\n{context}\n\nWorkspace files:\n{workspace_context}\n\nGoal: {goal}\n"
        self.logger.info("planning", goal=goal[:100])
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            plan = Plan(**data)
            self.logger.info("plan_created", task_count=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error("plan_parse_failed", error=str(e), response=response[:200])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    async def _get_workspace_context(self) -> str:
        """Get a summary of the workspace files for planning context."""
        try:
            workspace = Path.cwd()
            files = []
            for f in workspace.rglob("*"):
                if f.is_file() and not any(part.startswith('.') for part in f.parts):
                    # Skip common ignore patterns
                    if any(skip in str(f) for skip in ['__pycache__', '.pyc', 'node_modules', '.git', 'venv', 'env']):
                        continue
                    rel = f.relative_to(workspace)
                    size = f.stat().st_size
                    files.append(f"{rel} ({size} bytes)")
            return "\n".join(files[:100]) if files else "No files in workspace"
        except Exception as e:
            self.logger.warning("workspace_context_failed", error=str(e))
            return "Could not read workspace"
