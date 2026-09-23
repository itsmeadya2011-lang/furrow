from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str) -> Plan:
        # Gather context about the current project
        context = await self._gather_context()
        
        prompt = f"""{PLANNER_PROMPT}

Current project context:
{context}

Goal: {goal}

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

If the goal is too large for one cycle, say so in rationale and break it into the most critical first slice."""
        response = await self._plan_with_retry(prompt)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    async def _gather_context(self) -> str:
        """Gather context about the current project state."""
        context_parts = []
        workspace = Path.cwd()
        
        try:
            files = self.client.list_files(workspace)
            context_parts.append(f"Workspace files: {', '.join(files[:50])}")
        except Exception:
            pass
            
        # Read key config files if they exist
        key_files = ["pyproject.toml", "package.json", "Cargo.toml", "go.mod", "requirements.txt", "README.md"]
        for key_file in key_files:
            path = workspace / key_file
            if path.exists():
                try:
                    content = await self.client.read_file(path)
                    context_parts.append(f"=== {key_file} ===\n{content[:2000]}")
                except Exception:
                    pass
                    
        return "\n\n".join(context_parts) if context_parts else "Empty workspace."

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    async def _plan_with_retry(self, prompt: str) -> str:
        return await self.client.complete(prompt, model=self.client.settings.planner_model)
