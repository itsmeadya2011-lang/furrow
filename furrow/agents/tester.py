from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # If tests passed, return success
        if "passed" in test_output.lower() or "ok" in test_output.lower():
            return TestResult(passed=True, summary="All tests passed", failures=[])

        # Tests failed - try to fix them
        # Get relevant files from tasks
        file_paths = set()
        for task in tasks:
            file_paths.update(task.files)

        # Read relevant files
        file_contents = {}
        for file_path in file_paths:
            full_path = self.settings.workspace / file_path
            if full_path.exists():
                try:
                    file_contents[file_path] = await self.client.read_file(full_path)
                except Exception:  # noqa: BLE001, S110
                    pass

        # Also check for test files
        test_files = list(self.settings.workspace.rglob("test*.py")) + list(self.settings.workspace.rglob("*test*.py"))
        for test_file in test_files[:10]:
            rel_path = test_file.relative_to(self.settings.workspace)
            if str(rel_path) not in file_contents:
                try:
                    file_contents[str(rel_path)] = await self.client.read_file(test_file)
                except Exception:  # noqa: BLE001, S110
                    pass

        context = f"Test output:\n{test_output}\n\n"
        if file_contents:
            context += "Relevant files:\n"
            for path, content in file_contents.items():
                context += f"\n--- {path} ---\n{content}\n"

        fix_prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\n{context}\n\nTests are failing. Analyze the failures and provide fixes.\nReturn JSON with this format:\n{{\n  \"fixes\": [\n    {{\n      \"file\": \"path/to/file.py\",\n      \"content\": \"fixed file content\"\n    }}\n  ],\n  \"summary\": \"Description of fixes applied\",\n  \"passed\": false\n}}"

        response = await self.client.complete(fix_prompt, model=self.client.settings.tester_model)

        try:
            data = json.loads(response)
            fixes = data.get("fixes", [])
            summary = data.get("summary", "Attempted fixes")
            passed = data.get("passed", False)

            # Apply fixes
            for fix in fixes:
                file_path = fix.get("file")
                content = fix.get("content", "")
                if file_path:
                    full_path = self.settings.workspace / file_path
                    await self.client.write_file(full_path, content)

            # Re-run tests to verify
            if fixes:
                test_output = await self._run_tests()
                if "passed" in test_output.lower() or "ok" in test_output.lower():
                    return TestResult(passed=True, summary=f"Fixed: {summary}", failures=[])

            return TestResult(passed=passed, summary=summary, failures=[test_output])
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed=False, summary="Failed to parse fix response", failures=[test_output])

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
            except Exception:  # noqa: BLE001, S112
                continue
        return "No test runner found."
