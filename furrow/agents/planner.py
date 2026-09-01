from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan, settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.workspace = Path(settings.workspace) if settings else Path.cwd()

    async def plan(self, goal: str) -> Plan:
        context = await self._gather_context()
        prompt = f"""{PLANNER_PROMPT}

## Goal
{goal}

## Workspace Context
{context}

Remember: Return only valid JSON with the plan structure."""

        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            # Try to extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()

            data = json.loads(json_str)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")

    async def _gather_context(self) -> str:
        """Gather workspace context for better planning."""
        context_parts = []

        # List workspace files
        files = self.client.list_files(self.workspace)
        if files:
            context_parts.append("### Project Structure")
            # Limit to first 100 files to avoid context overflow
            for f in sorted(files)[:100]:
                context_parts.append(f"- {f}")
            if len(files) > 100:
                context_parts.append(f"\n... and {len(files) - 100} more files")

        # Try to read key configuration files
        config_files = [
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "README.md",
            "setup.py",
            "requirements.txt",
        ]

        for config_file in config_files:
            config_path = self.workspace / config_file
            if config_path.exists():
                try:
                    content = await self.client.read_file(config_path)
                    # Limit content length
                    if len(content) > 2000:
                        content = content[:2000] + "\n... (truncated)"
                    context_parts.append(f"\n### {config_file}\n```\n{content}\n```")
                except Exception:
                    pass  # Skip files that can't be read

        return "\n".join(context_parts) if context_parts else "[No workspace context available]"
