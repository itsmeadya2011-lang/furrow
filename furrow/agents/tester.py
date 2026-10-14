from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except OSError as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q"],
            ["python", "-m", "pytest", "-q"],
            ["npm", "test", "--", "--silent"],
            ["pnpm", "test", "--", "--silent"],
            ["yarn", "test", "--silent"],
            ["cargo", "test", "-q"],
            ["go", "test", "./..."],
        ]
        for cmd in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except FileNotFoundError:
                continue
        return "No test runner found."

    async def fix_failures(
        self, goal: str, failures: list[str], tasks: list[TaskModel]
    ) -> list[TaskModel]:
        """Generate fix tasks for test failures."""
        if not failures:
            return []

        fix_prompt = f"""Tests failed. Create tasks to fix these failures.

Goal: {goal}
Failures:
{chr(10).join(f"  - {f}" for f in failures)}

Return JSON with tasks to fix the failures:
{{
  "tasks": [
    {{"id": "fix-1", "description": "...", "files": ["..."], "dependencies": []}}
  ],
  "rationale": "..."
}}"""

        from furrow.agents.planner import PlannerAgent

        PlannerAgent(client=self.client)
        # We'll create a simple plan for fixes
        response = await self.client.complete(fix_prompt, model=self.client.settings.planner_model)
        try:
            data = json.loads(response)
            fix_tasks = []
            for i, task_data in enumerate(data.get("tasks", [])):
                fix_tasks.append(
                    TaskModel(
                        id=task_data.get("id", f"fix-{i + 1}"),
                        description=task_data.get("description", ""),
                        files=task_data.get("files", []),
                        dependencies=task_data.get("dependencies", []),
                    )
                )
            return fix_tasks
        except (json.JSONDecodeError, ValueError):
            # Fallback: create a generic fix task
            return [
                TaskModel(
                    id="fix-1",
                    description=f"Fix test failures: {', '.join(failures[:3])}",
                    files=[t.files[0] for t in tasks if t.files],
                    dependencies=[],
                )
            ]
