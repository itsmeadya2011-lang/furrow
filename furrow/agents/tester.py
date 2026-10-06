from __future__ import annotations

import asyncio
import json
import os
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_runner_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        logger.debug("test_output", output=test_output[:500])
        parsed = self._parse_test_output(test_output)
        
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model, json_mode=True)
        logger.debug("tester_response", response=response)
        try:
            data = json.loads(response)
            result = TestResult(**data)
            # Merge with parsed results
            if not result.passed and parsed["failures"]:
                result.failures.extend(parsed["failures"])
            logger.info("tester_parsed", passed=result.passed, failures=len(result.failures))
            return result
        except (json.JSONDecodeError, ValueError):
            passed = "passed" in response.lower() and "failed" not in response.lower()
            logger.warning("tester_parse_failed", passed=passed)
            return TestResult(passed=passed, summary=response, failures=parsed["failures"])

    def _parse_test_output(self, output: str) -> dict:
        """Parse common test output formats to extract failures."""
        failures = []
        
        # pytest failures
        pytest_failures = re.findall(r"FAILED\s+(.+?)(?:\s+-\s+|\n)", output)
        for f in pytest_failures:
            failures.append(f"pytest: {f.strip()}")
        
        # Generic error patterns
        error_patterns = [
            r"(Error|Exception|Traceback):\s*(.+)",
            r"FAILED:\s*(.+)",
            r"✗\s*(.+)",
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, output, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    failures.append(f"{match[0]}: {match[1]}")
                else:
                    failures.append(match)
        
        return {"failures": list(set(failures))}

    async def _run_tests(self) -> str:
        candidates = [
            ["pytest", "-q", "--tb=short"],
            ["python", "-m", "pytest", "-q", "--tb=short"],
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
                    result = stdout.decode() + stderr.decode()
                    logger.info("tests_run", command=cmd[0], returncode=proc.returncode)
                    return result
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("test_timeout", command=cmd[0])
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_runner_not_found", command=cmd[0], error=str(e))
                continue
        return "No test runner found."
