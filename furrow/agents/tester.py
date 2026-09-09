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
        test_framework = ""
        try:
            test_output, test_framework = await self._run_tests()
        except Exception as e:
            logger.error("test_runner_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest framework: {test_framework}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: try to parse pass/fail from output
            passed = self._parse_test_result(test_output, test_framework)
            return TestResult(
                passed=passed,
                summary=test_output[:500] if test_output else "No test output",
                failures=[] if passed else [test_output[:1000]]
            )

    async def _run_tests(self) -> tuple[str, str]:
        candidates = [
            (["pytest", "-q", "--tb=short"], "pytest"),
            (["python", "-m", "pytest", "-q", "--tb=short"], "pytest"),
            (["npm", "test", "--", "--silent"], "jest"),
            (["pnpm", "test", "--", "--silent"], "jest"),
            (["yarn", "test", "--silent"], "jest"),
            (["cargo", "test", "-q"], "cargo"),
            (["go", "test", "./..."], "go"),
        ]
        for cmd, framework in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return output, framework
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, Exception) as e:
                logger.debug("test_command_failed", cmd=cmd, error=str(e))
                continue
        return "No test runner found or no tests discovered.", "unknown"

    def _parse_test_result(self, output: str, framework: str) -> bool:
        if not output.strip():
            return False
            
        output_lower = output.lower()
        
        if framework == "pytest":
            # Look for pytest patterns
            if "failed" in output_lower or "error" in output_lower:
                # Check if there are actual failures vs just warnings
                if re.search(r'\d+\s+failed', output_lower):
                    return False
                if re.search(r'\d+\s+error', output_lower):
                    return False
            if "passed" in output_lower and "failed" not in output_lower and "error" not in output_lower:
                return True
                
        elif framework in ("jest", "vitest"):
            if "test suites.*failed" in output_lower or "tests.*failed" in output_lower:
                return False
            if "all tests passed" in output_lower or "test suites.*passed" in output_lower:
                return True
                
        elif framework == "cargo":
            if "test result: ok" in output_lower:
                return True
            if "test result: failed" in output_lower:
                return False
                
        elif framework == "go":
            if "FAIL" in output:
                return False
            if "PASS" in output and "FAIL" not in output:
                return True
        
        # Default: check for common failure indicators
        failure_indicators = ["failed", "error", "fail:", "assertion error", "traceback"]
        success_indicators = ["passed", "ok.", "all tests passed", "success"]
        
        for indicator in failure_indicators:
            if indicator in output_lower:
                return False
        for indicator in success_indicators:
            if indicator in output_lower:
                return True
                
        return False
