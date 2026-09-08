from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.agents.prompts import TESTER_PROMPT
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

logger = structlog.get_logger(__name__)


class TesterAgent:
    """Agent that runs tests and attempts to fix failures."""

    # Map of file patterns to test commands for smarter discovery
    TEST_RUNNERS: list[tuple[str, list[str]]] = [
        ("pytest.ini", ["pytest", "-q"]),
        ("setup.py", ["pytest", "-q"]),
        ("pyproject.toml", ["python", "-m", "pytest", "-q"]),
        ("package.json", ["npm", "test", "--", "--silent"]),
        ("Cargo.toml", ["cargo", "test", "-q"]),
        ("go.mod", ["go", "test", "./..."]),
    ]

    def __init__(
        self,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run tests, diagnose failures, and attempt fixes.

        If tests fail, a WorkerAgent is spawned to fix the issues.
        After fixing, tests are re-run once to verify.
        """
        logger.info("Running tests", goal=goal)
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("Test execution failed", error=str(e))
            return TestResult(passed=False, summary=f"Test execution error: {e}", failures=[str(e)])

        if not test_output:
            test_output = "No test output"

        result = await self._analyze_test_output(goal, tasks, test_output)

        # If tests failed, attempt a fix
        if not result.passed and result.failures:
            logger.info("Tests failed, attempting fix", failures=result.failures)
            fix_result = await self._fix_failures(goal, tasks, result)
            if fix_result:
                # Re-run tests after fix
                test_output_after = await self._run_tests()
                result_after = await self._analyze_test_output(goal, tasks, test_output_after)
                if result_after.passed:
                    logger.info("Fix successful")
                    return TestResult(
                        passed=True,
                        summary=f"Fix successful: {result_after.summary}",
                        failures=[],
                    )
                else:
                    logger.warning("Fix did not resolve all failures")
                    return result_after

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _analyze_test_output(
        self, goal: str, tasks: list[TaskModel], test_output: str
    ) -> TestResult:
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(
            prompt,
            model=self.client.settings.tester_model,
        )
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            # Fallback heuristic
            passed = "all tests passed" in response.lower() or "passed" in response.lower() and "failed" not in response.lower()
            return TestResult(
                passed=passed,
                summary=response[:500],
                failures=[] if passed else [response[:500]],
            )

    async def _fix_failures(
        self, goal: str, tasks: list[TaskModel], result: TestResult
    ) -> bool:
        """Attempt to fix test failures using a WorkerAgent."""
        failures_str = "\n".join(result.failures)
        fix_task = TaskModel(
            id="fix-tests",
            description=(
                f"Fix the following test failures:\n{failures_str}\n\n"
                f"Original goal: {goal}\n"
                "Make minimal changes to fix the failing tests."
            ),
            files=[],
            dependencies=[],
        )
        try:
            worker = WorkerAgent(task=fix_task, client=self.client, settings=self.client.settings)
            summary = await worker.run()
            logger.info("Fix worker completed", summary=summary[:200])
            return True
        except Exception as e:
            logger.error("Fix worker failed", error=str(e))
            return False

    async def _run_tests(self) -> str:
        """Run the project's test suite.

        Tries to detect the appropriate test runner based on project files,
        falling back to a list of common runners.
        """
        workspace = self.client.settings.workspace
        commands = self._detect_test_commands(workspace)

        for cmd in commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode() + stderr.decode()
                    if proc.returncode == 0:
                        logger.debug("Test runner succeeded", command=cmd)
                        return output
                except asyncio.TimeoutError:
                    proc.kill()
                    logger.warning("Test command timed out", command=cmd)
                    continue
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.warning("Test command error", command=cmd, error=str(e))
                continue

        # No test runner found
        logger.info("No test runner found")
        return "No test runner found."

    def _detect_test_commands(self, workspace: os.PathLike | str) -> list[list[str]]:
        """Detect appropriate test commands based on project files in the workspace."""
        import pathlib

        ws = pathlib.Path(workspace)
        commands: list[list[str]] = []

        for marker, cmd in self.TEST_RUNNERS:
            if (ws / marker).exists():
                commands.append(cmd)

        # Also check for Makefile with test target
        if (ws / "Makefile").exists():
            commands.append(["make", "test"])

        # Fallback to common runners if no project files detected
        if not commands:
            commands = [
                ["python", "-m", "pytest", "-q"],
                ["npm", "test", "--", "--silent"],
                ["cargo", "test", "-q"],
                ["go", "test", "./..."],
            ]

        return commands
