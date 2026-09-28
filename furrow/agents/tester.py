from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from structlog import get_logger

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger()


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:  # noqa: BLE001
            logger.error("test_runner_exception", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        return self._parse_tester_response(response)

    def _parse_tester_response(self, response: str) -> TestResult:
        stripped = response.strip()
        stripped = stripped.removeprefix("```json")
        stripped = stripped.removesuffix("```")
        stripped = stripped.strip()

        try:
            data = json.loads(stripped)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            logger.warning("tester_json_parse_failed", response=stripped[:200])
            # Fallback: use substring matching to determine pass/fail
            has_failures = any(marker in response.lower() for marker in ["fail", "error"])
            passed = "passed" in response.lower() and not has_failures
            return TestResult(
                passed=passed,
                summary=response,
                failures=[] if passed else [response],
            )

    async def _run_tests(self) -> str:
        """Run tests, linting, and type checking, returning combined output."""
        commands = [
            ("tests", ["pytest", "-q"]),
            ("tests", ["python", "-m", "pytest", "-q"]),
            ("tests", ["npm", "test", "--", "--silent"]),
            ("tests", ["pnpm", "test", "--", "--silent"]),
            ("tests", ["yarn", "test", "--silent"]),
            ("tests", ["cargo", "test", "-q"]),
            ("tests", ["go", "test", "./..."]),
        ]

        output_parts: list[str] = []
        test_ran = False

        for label, cmd in commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    decoded = stdout.decode() + stderr.decode()
                    if proc.returncode == 0 and decoded.strip():
                        output_parts.append(f"[{label} ({' '.join(cmd)})] exit_code=0\n{decoded}")
                        test_ran = True
                    elif proc.returncode != 0 and decoded.strip():
                        output_parts.append(f"[{label} ({' '.join(cmd)})] exit_code={proc.returncode}\n{decoded}")
                        test_ran = True
                except asyncio.TimeoutError:
                    proc.kill()
                    output_parts.append(f"[{label} ({' '.join(cmd)})] timed out after 120s")
                    test_ran = True
            except (FileNotFoundError, Exception):  # noqa: BLE001
                logger.debug("command_not_found", command=" ".join(cmd))
                continue

        # Run linters if available
        lint_commands = [
            ("lint (ruff)", ["ruff", "check", "."]),
            ("lint (flake8)", ["flake8", "."]),
            ("types (mypy)", ["mypy", "."]),
        ]
        for label, cmd in lint_commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                decoded = stdout.decode() + stderr.decode()
                if decoded.strip():
                    status = "OK" if proc.returncode == 0 else f"exit_code={proc.returncode}"
                    output_parts.append(f"[{label}] {status}\n{decoded}")
            except (FileNotFoundError, asyncio.TimeoutError, Exception):  # noqa: BLE001
                logger.debug("lint_command_skipped", command=" ".join(cmd))
                continue

        if not test_ran:
            output_parts.append("No test runner found.")

        return "\n\n".join(output_parts)
