from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.config import get_logger
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except (OSError, asyncio.SubprocessError) as e:
            logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # If tests passed, return success
        if "passed" in test_output.lower() and "failed" not in test_output.lower():
            logger.info("tests_passed", output=test_output[:200])
            return TestResult(passed=True, summary="All tests passed", failures=[])

        # Tests failed - try to fix them
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        
        try:
            data = json.loads(response)
            passed = data.get("passed", False)
            summary = data.get("summary", response)
            failures = data.get("failures", [])
            
            # If LLM says it can fix, try to apply fixes
            if not passed and "fixes" in data:
                for fix in data["fixes"]:
                    path = fix.get("path")
                    content = fix.get("content", "")
                    if path and content:
                        await self.client.write_file(path, content)
                        logger.info("fix_applied", path=path)
                
                # Re-run tests after fixes
                test_output = await self._run_tests()
                if "passed" in test_output.lower() and "failed" not in test_output.lower():
                    logger.info("tests_passed_after_fix")
                    return TestResult(passed=True, summary="Tests passed after fixes", failures=[])
            
            logger.info("test_result", passed=passed, failures=len(failures))
            return TestResult(passed=passed, summary=summary, failures=failures)
        except (json.JSONDecodeError, ValueError):
            # Fallback: simple pass/fail detection
            passed = "passed" in response.lower() and "failed" not in response.lower()
            logger.warning("non_json_test_response", passed=passed)
            return TestResult(passed=passed, summary=response, failures=[] if passed else [response])

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
            except OSError:
                continue
        return "No test runner found."
