from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        from furrow.config import settings as global_settings

        self.client = client or LLMClient(settings=settings or global_settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except (RuntimeError, OSError, asyncio.TimeoutError) as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        # Try to parse test results directly first
        parsed = self._parse_test_output(test_output)
        if parsed is not None:
            return parsed

        # Fallback to LLM interpretation
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

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
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, RuntimeError, OSError):
                continue
        return "No test runner found."

    def _parse_test_output(self, output: str) -> TestResult | None:
        """Parse test output from common test frameworks."""
        if not output or output == "No test runner found.":
            return None

        try:
            # pytest
            if "pytest" in output or "FAILED" in output or "PASSED" in output or "ERROR" in output:
                return self._parse_pytest(output)

            # npm/jest
            if ("npm test" in output or "PASS" in output or "FAIL" in output) and ("Test Suites:" in output or "Tests:" in output):
                return self._parse_jest(output)

            # cargo
            if "cargo test" in output or "test result:" in output:
                return self._parse_cargo(output)

            # go test
            if ("go test" in output or "PASS" in output or "FAIL" in output) and ("ok\t" in output or "FAIL\t" in output):
                return self._parse_go_test(output)
        except Exception:
            return None

        return None

    def _parse_pytest(self, output: str) -> TestResult:
        # Look for pytest summary line: "X passed, Y failed, Z error"
        passed = failed = errors = 0
        failures = []

        # Check for short summary
        short_match = re.search(
            r"=+\s*(\d+ passed)(?:, (\d+ failed))?(?:, (\d+ error))?(?:, (\d+ skipped))?\s*=+",
            output,
        )
        if short_match:
            passed = int(short_match.group(1).split()[0]) if short_match.group(1) else 0
            failed = int(short_match.group(2).split()[0]) if short_match.group(2) else 0
            errors = int(short_match.group(3).split()[0]) if short_match.group(3) else 0

        # Extract failure details
        failure_sections = re.findall(r"(FAILED|ERROR)\s+([^\n]+)", output)
        for kind, test_name in failure_sections:
            failures.append(f"{kind}: {test_name.strip()}")

        # Also check for traceback sections
        tracebacks = re.findall(r"_{10,}\s+(FAILED|ERROR)\s+([^\n]+)", output)
        for kind, test_name in tracebacks:
            if f"{kind}: {test_name.strip()}" not in failures:
                failures.append(f"{kind}: {test_name.strip()}")

        total_failed = failed + errors
        passed_tests = passed
        total = passed_tests + total_failed

        if total == 0:
            # Could not parse, return None to use LLM
            raise ValueError("Could not parse pytest output")

        return TestResult(
            passed=total_failed == 0,
            summary=f"{passed_tests} passed, {total_failed} failed"
            if total > 0
            else "No tests run",
            failures=failures,
        )

    def _parse_jest(self, output: str) -> TestResult:
        # Jest output parsing
        passed = failed = 0
        failures = []

        # Look for "Tests: X passed, Y failed"
        match = re.search(r"Tests:\s+(\d+)\s+passed(?:,\s+(\d+)\s+failed)?", output)
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2)) if match.group(2) else 0

        # Extract failed test names
        failed_tests = re.findall(r"●\s+(.+?)\n", output)
        for test in failed_tests:
            failures.append(test.strip())

        return TestResult(
            passed=failed == 0,
            summary=f"{passed} passed, {failed} failed",
            failures=failures,
        )

    def _parse_cargo(self, output: str) -> TestResult:
        # cargo test output: "test result: ok. X passed; Y failed"
        match = re.search(
            r"test result:\s+(ok|FAILED)\.\s+(\d+)\s+passed(?:;\s+(\d+)\s+failed)?", output
        )
        if match:
            passed = int(match.group(2))
            failed = int(match.group(3)) if match.group(3) else 0
            return TestResult(
                passed=match.group(1) == "ok",
                summary=f"{passed} passed, {failed} failed",
                failures=[],
            )
        raise ValueError("Could not parse cargo test output")

    def _parse_go_test(self, output: str) -> TestResult:
        # go test output: "ok\tpackage\t0.123s" or "FAIL\tpackage\t0.123s"
        lines = output.strip().split("\n")
        passed_packages = 0
        failed_packages = 0
        failures = []

        for line in lines:
            if line.startswith("ok\t"):
                passed_packages += 1
            elif line.startswith("FAIL\t"):
                failed_packages += 1
                failures.append(line.strip())

        return TestResult(
            passed=failed_packages == 0,
            summary=f"{passed_packages} packages passed, {failed_packages} failed",
            failures=failures,
        )
