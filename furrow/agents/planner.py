from __future__ import annotations

import json
import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings, TaskModel

logger = structlog.get_logger()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, previous_tasks: list | None = None) -> Plan:
        # Get codebase context
        codebase_context = await self._get_codebase_context()
        
        task_context = ""
        if previous_tasks:
            task_context = "\n\nPrevious tasks:\n"
            for t in previous_tasks:
                task_context += f"- {t.id}: {t.description} [{t.status}]\n"
                if t.result:
                    task_context += f"  Result: {t.result[:200]}\n"

        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n\nCodebase context:\n{codebase_context}{task_context}\n"
        logger.debug("planner_prompt", prompt=prompt[:500])
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        logger.debug("planner_response", response=response[:500])
        try:
            data = json.loads(response)
            plan = Plan(**data)
            logger.info("plan_created", num_tasks=len(plan.tasks))
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("plan_parse_failed", error=str(e), response=response[:500])
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    async def _get_codebase_context(self) -> str:
        """Get relevant codebase context for planning."""
        try:
            files = self.client.list_files(self.client.settings.workspace)
            # Filter to relevant files (source code, not build artifacts)
            relevant_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java', '.cpp', '.c', '.h', '.cs', '.rb', '.php', '.swift', '.kt', '.scala', '.clj', '.hs', '.ml', '.fs', '.vue', '.svelte', '.html', '.css', '.scss', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.md', '.txt', '.rst'}
            relevant_files = [
                f for f in files 
                if any(f.endswith(ext) for ext in relevant_extensions)
                and not any(skip in f for skip in ['__pycache__', 'node_modules', '.git', 'dist', 'build', 'target', '.venv', 'venv', 'env'])
            ][:50]  # Limit to 50 files
            
            if not relevant_files:
                return "No relevant source files found."
            
            # Read first few files to get context
            context_parts = [f"Project files ({len(relevant_files)} total):"]
            for f in relevant_files[:10]:
                try:
                    content = await self.client.read_file(self.client.settings.workspace / f)
                    context_parts.append(f"\n--- {f} ---\n{content[:1000]}")
                except Exception:
                    pass
            
            return "\n".join(context_parts)
        except Exception as e:
            logger.warning("codebase_context_failed", error=str(e))
            return f"Error getting codebase context: {e}"
