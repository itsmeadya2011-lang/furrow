from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def plan(self, goal: str, context: str = "") -> Plan:
        prompt = f"{PLANNER_PROMPT}\n\nGoal: {goal}\n"
        if context:
            prompt += f"\n{context}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    async def get_project_context(self, workspace: str = ".") -> str:
        """Get relevant project context by reading key files."""
        try:
            files = self.client.list_files(workspace)
            # Filter for relevant files
            relevant = [
                f for f in files
                if any(f.endswith(ext) for ext in [".py", ".js", ".ts", ".json", ".toml", ".yaml", ".yml", ".md"])
                and not any(skip in f for skip in ["__pycache__", ".git", "node_modules", ".venv", "venv"])
            ][:20]  # Limit to 20 files
            
            if not relevant:
                return ""
            
            # Read a few key files
            file_contents = {}
            for f in relevant[:5]:
                try:
                    content = await self.client.read_file(f)
                    file_contents[f] = content[:2000]  # Limit content
                except Exception:
                    pass
            
            if not file_contents:
                return ""
            
            parts = ["## Project Context"]
            for path, content in file_contents.items():
                parts.append(f"\n### {path}\n```\n{content}\n```")
            return "\n".join(parts)
        except Exception:
            return ""
