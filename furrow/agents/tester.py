from __future__ import annotations

import asyncio
import json
import os
import structlog
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_run_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        logger.debug("tester_prompt", prompt=prompt[:500])
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        logger.debug("tester_response", response=response[:500])
        try:
            data = json.loads(response)
            result = TestResult(**data)
            logger.info("tester_result", passed=result.passed, summary=result.summary)
            
            # If tests failed, try to fix them
            if not result.passed and result.failures:
                await self._attempt_fix(goal, result.failures, test_output)
            
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower()
            result = TestResult(passed=passed, summary=response, failures=[] if passed else [response])
            logger.warning("tester_parse_failed", passed=passed, response=response[:500])
            return result

    async def _attempt_fix(self, goal: str, failures: list[str], test_output: str) -> None:
        """Attempt to fix failing tests by creating fix tasks."""
        fix_prompt = f"""You are a test fixing agent in Furrow.

Given the goal, test failures, and test output, create a plan to fix the issues.

Return JSON only with this shape:
{{
  "fixes": [
    {{
      "file": "path/to/file.py",
      "description": "What to fix",
      "content": "full fixed file content"
    }}
  ]
}}

Goal: {goal}

Test failures:
{chr(10).join(failures)}

Test output:
{test_output[:3000]}
"""
        try:
            response = await self.client.complete(fix_prompt, model=self.client.settings.tester_model)
            data = json.loads(response)
            
            if "fixes" in data:
                for fix in data["fixes"]:
                    file_path = fix.get("file", "")
                    content = fix.get("content", "")
                    if file_path and content:
                        await self.client.write_file(file_path, content)
                        logger.info("test_fix_applied", file=file_path)
        except Exception as e:
            logger.error("test_fix_failed", error=str(e))

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
                    output = stdout.decode() + stderr.decode()
                    logger.debug("test_command_output", cmd=cmd[0], output=output[:500])
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", cmd=cmd[0])
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_command_not_found", cmd=cmd[0], error=str(e))
                continue
        logger.warning("no_test_runner_found")
        return "No test runner found."
