from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = logging.getLogger(__name__)


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        # Run all checks and collect output
        all_output = []

        # Run tests
        test_output = await self._run_tests()
        if test_output and test_output != "No test runner found.":
            all_output.append(f"=== TESTS ===\n{test_output}")

        # Run linting
        lint_output = await self._run_lint()
        if lint_output:
            all_output.append(f"=== LINT ===\n{lint_output}")

        # Run type checking
        type_output = await self._run_type_check()
        if type_output:
            all_output.append(f"=== TYPE CHECK ===\n{type_output}")

        combined_output = "\n\n".join(all_output) if all_output else "No checks configured or found."

        prompt = TESTER_PROMPT.format(
            goal=goal,
            test_output=combined_output,
        )
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback: parse manually
            passed = self._parse_passed(combined_output, response)
            failures = self._extract_failures(combined_output)
            return TestResult(passed=passed, summary=response, failures=failures)

    def _parse_passed(self, output: str, response: str) -> bool:
        """Determine if checks passed from output and LLM response."""
        if "passed" in response.lower() and "failed" not in response.lower():
            return True
        # Check for common failure patterns in output
        failure_patterns = [
            r"FAILED",
            r"ERROR",
            r"failed",
            r"error:",
            r"✗",
            r"❌",
        ]
        for pattern in failure_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                return False
        return "passed" in output.lower() or "ok" in output.lower()

    def _extract_failures(self, output: str) -> list[str]:
        """Extract failure messages from output."""
        failures = []
        lines = output.split("\n")

        # Common failure patterns
        for i, line in enumerate(lines):
            if any(p in line.lower() for p in ["failed", "error:", "assertion", "traceback"]):
                # Include context
                context = "\n".join(lines[max(0, i-2):min(len(lines), i+3)])
                failures.append(context.strip())
                if len(failures) >= 10:
                    break

        return failures[:10]

    async def _run_tests(self) -> str:
        candidates = [
            (["pytest", "-q", "--tb=short"], "pytest"),
            (["python", "-m", "pytest", "-q", "--tb=short"], "pytest"),
            (["npm", "test", "--", "--silent"], "npm"),
            (["pnpm", "test", "--", "--silent"], "pnpm"),
            (["yarn", "test", "--silent"], "yarn"),
            (["cargo", "test", "-q"], "cargo"),
            (["go", "test", "./..."], "go"),
        ]
        for cmd, name in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return f"[{name}]\n{output}"
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except OSError:
                logger.debug("Command %s not available, skipping", cmd[0])
                continue
        return ""

    async def _run_lint(self) -> str:
        candidates = [
            (["ruff", "check", "."], "ruff"),
            (["python", "-m", "ruff", "check", "."], "ruff"),
            (["eslint", "."], "eslint"),
            (["npx", "eslint", "."], "eslint"),
            (["golangci-lint", "run"], "golangci-lint"),
        ]
        for cmd, name in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return f"[{name}]\n{output}"
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except OSError:
                logger.debug("Command %s not available, skipping", cmd[0])
                continue
        return ""

    async def _run_type_check(self) -> str:
        candidates = [
            (["mypy", "."], "mypy"),
            (["python", "-m", "mypy", "."], "mypy"),
            (["tsc", "--noEmit"], "tsc"),
            (["npx", "tsc", "--noEmit"], "tsc"),
            (["go", "vet", "./..."], "go vet"),
        ]
        for cmd, name in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if output.strip():
                        return f"[{name}]\n{output}"
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except OSError:
                logger.debug("Command %s not available, skipping", cmd[0])
                continue
        return ""