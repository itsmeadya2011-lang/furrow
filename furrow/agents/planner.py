from __future__ import annotations

import json
from pathlib import Path

from furrow.agents.prompts import PLANNER_PROMPT, PLANNER_SYSTEM
from furrow.config import Plan, Settings
from furrow.llm import LLMClient
from furrow.logging import get_logger

log = get_logger(__name__)


class PlannerAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self.workspace = workspace or self.client.settings.workspace
        self.settings = self.client.settings

    async def plan(self, goal: str) -> Plan:
        context = await self._gather_context()
        prompt = PLANNER_PROMPT.replace("{context}", context) + f"\n\nGoal: {goal}\n"
        log.debug("planner starting", goal=goal)
        response = await self.client.complete(
            prompt, system=PLANNER_SYSTEM, model=self.client.settings.planner_model
        )
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            log.error("failed to parse plan", error=str(e), response=response[:500])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    async def _gather_context(self, max_files: int = 30) -> str:
        if not self.workspace.exists():
            return ""
        files = sorted(
            f for f in self.workspace.rglob("*") if f.is_file() and self._is_readable(f)
        )[:max_files]
        parts: list[str] = []
        for f in files:
            try:
                content = await self.client.read_file(f)
                parts.append(f"# {f.relative_to(self.workspace)}\n```\n{content[:2000]}\n```\n")
            except Exception:
                pass
        return "\n".join(parts)

    @staticmethod
    def _is_readable(path: Path) -> bool:
        return path.suffix not in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".so", ".pdf", ".zip"}
