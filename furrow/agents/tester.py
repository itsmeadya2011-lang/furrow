from __future__ import annotations

import asyncio
import json

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings or Settings())
        self.settings = settings or self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # If tests passed, return success
        if "passed" in test_output.lower() and "failed" not in test_output.lower() and "error" not in test_output.lower():
            return TestResult(passed=True, summary=test_output, failures=[])

        # Tests failed - try to fix them
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        
        try:
            data = json.loads(response)
            # If the tester returned fix actions, execute them
            if "fix" in data:
                await self._execute_fixes(data["fix"])
                # Re-run tests after fixes
                test_output = await self._run_tests()
                if "passed" in test_output.lower() and "failed" not in test_output.lower():
                    return TestResult(passed=True, summary="Fixed: " + test_output, failures=[])
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
            except (FileNotFoundError, OSError):
                continue
        return "No test runner found."

    async def _execute_fixes(self, fixes: list[dict]) -> None:
        """Execute fix actions returned by the LLM."""
        for fix in fixes:
            action = fix.get("action", "")
            path = fix.get("path", "")
            try:
                if action == "write":
                    content = fix.get("content", "")
                    await self.client.write_file(path, content)
                elif action == "edit":
                    old_str = fix.get("old_str", "")
                    new_str = fix.get("new_str", "")
                    content = await self.client.read_file(path)
                    if old_str in content:
                        new_content = content.replace(old_str, new_str)
                        await self.client.write_file(path, new_content)
            except OSError:
                pass  # Silently continue
