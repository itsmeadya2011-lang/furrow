from __future__ import annotations

import asyncio
import json
import shutil
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import logger

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    """Agent that runs the project's test suite and judges pass/fail via LLM."""

    # Timeout for individual test commands (seconds)
    TEST_TIMEOUT = 120

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        log = logger.bind(goal=goal, task_count=len(tasks))
        log.debug("tester_running")
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            log.warning("test_execution_error", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        log.debug("test_output_received", length=len(test_output))

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        try:
            data = json.loads(response)
            result = TestResult(**data)
            log.info("test_judgment_complete", passed=result.passed)
            return result
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: keyword-based judgment
            passed = "passed" in response.lower() and "failed" not in response.lower()
            log.warning("json_parse_failed_fallback", error=str(e), passed=passed)
            return TestResult(passed=passed, summary=response, failures=[])

    async def _run_tests(self) -> str:
        """Run the project's test runner, linter, and type checker in the workspace."""
        workspace = self.client.settings.workspace
        outputs: list[str] = []

        # Determine which test runners and tools are available
        test_commands = self._detect_test_commands()

        if not test_commands:
            return "No test runner found."

        for cmd in test_commands:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(workspace) if workspace.exists() else None,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=self.TEST_TIMEOUT
                    )
                    outputs.append(f"$ {' '.join(cmd)}\n{stdout.decode()}\n{stderr.decode()}")
                except asyncio.TimeoutError:
                    proc.kill()
                    outputs.append(f"$ {' '.join(cmd)}\n[TIMEOUT after {self.TEST_TIMEOUT}s]")
            except (FileNotFoundError, Exception) as e:
                outputs.append(f"$ {' '.join(cmd)}\n[Command not found: {e}]")

        return "\n\n---\n\n".join(outputs)

    def _detect_test_commands(self) -> list[list[str]]:
        """Detect which test runners and quality tools are available in the workspace."""
        commands: list[list[str]] = []

        # Test runners (try in order of likelihood)
        if shutil.which("pytest"):
            commands.append(["pytest", "-q"])
        if shutil.which("ruff"):
            commands.append(["ruff", "check", "."])
        if shutil.which("mypy"):
            commands.append(["mypy", "."])

        return commands
