from __future__ import annotations

import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import PLANNER_PROMPT
from furrow.config import Plan
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

KEY_FILE_NAMES = ("pyproject.toml", "README.md", "README.txt", "setup.py", "setup.cfg", "package.json", "Cargo.toml", "go.mod", "main.py", "app.py", "index.py", "src/index.ts", "src/main.ts", "src/main.py")


class PlannerAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def _gather_context(self) -> str:
        workspace = self.client.settings.workspace
        files = self.client.list_files(workspace)
        file_list = "\n".join(files) if files else "(empty workspace)"

        key_files: list[str] = []
        for name in KEY_FILE_NAMES:
            for f in files:
                if f.endswith(name) or f == name:
                    key_files.append(f)
            if len(key_files) >= 5:
                break

        contents = ""
        for f in key_files[:5]:
            try:
                content = await self.client.read_file(workspace / f)
                contents += f"\n\n--- {f} ---\n{content}\n"
            except Exception:
                continue

        return f"## Existing files\n{file_list}\n\n## Key file contents\n{contents or '(none)'}\n"

    async def plan(self, goal: str) -> Plan:
        context = await self._gather_context()
        prompt = f"{PLANNER_PROMPT}\n\n{context}\n\nGoal: {goal}\n"
        response = await self.client.complete(prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            return Plan(**data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse plan from LLM: {e}\nResponse: {response}")
