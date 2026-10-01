from __future__ import annotations

import asyncio
import json
import shutil
from typing import TYPE_CHECKING

from furrow.agents.prompts import render_tester_prompt
from furrow.config import TaskModel, TestResult, get_logger
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

log = get_logger("furrow.tester")

# Test runner candidates: (command, name)
TEST_RUNNERS: list[tuple[list[str], str]] = [
    (["pytest", "-q"], "pytest"),
    (["python", "-m", "pytest", "-q"], "python -m pytest"),
    (["npm", "test", "--", "--silent"], "npm"),
    (["pnpm", "test", "--", "--silent"], "pnpm"),
    (["yarn", "test", "--silent"], "yarn"),
    (["cargo", "test", "-q"], "cargo"),
    (["go", "test", "./..."], "go"),
    (["dotnet", "test"], "dotnet"),
    (["mix", "test"], "mix"),
    (["bundle", "exec", "rspec"], "rspec"),
]


class TesterAgent:
    """Agent that runs tests and evaluates results."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Run tests and evaluate results against the goal."""
        log.info("tester.run.start", goal=goal, num_tasks=len(tasks))

        try:
            test_output = await self._run_tests()
        except Exception as e:
            log.error("tester.run.test_error", error=str(e))
            return TestResult(
                passed=False,
                summary=f"Failed to run tests: {e}",
                failures=[f"Test runner error: {e}"],
            )

        if test_output == "No test runner found.":
            log.info("tester.run.no_runner")
            return TestResult(
                passed=True,
                summary="No test runner found. Unable to verify tests.",
                failures=[],
            )

        task_descriptions = ", ".join(t.description for t in tasks)
        prompt = render_tester_prompt(test_output, goal, task_descriptions)

        response = await self.client.complete(prompt, model=self.settings.tester_model)

        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            lower = response.lower()
            has_passed = "passed" in lower or "pass" in lower
            has_failures = "fail" in lower
            passed = has_passed and not has_failures
            return TestResult(
                passed=passed,
                summary=response[:500],
                failures=[] if passed else [response[:500]],
            )

    async def _run_tests(self) -> str:
        """Run tests using the first available test runner."""
        for cmd, name in TEST_RUNNERS:
            executable = cmd[0]
            if not shutil.which(executable):
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.settings.workspace),
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
                    log.info("tester.run.completed", runner=name, return_code=proc.returncode)
                    return output
                except asyncio.TimeoutError:
                    proc.kill()
                    log.warning("tester.run.timeout", runner=name)
                    return f"Test runner '{name}' timed out after 120s.\n"
            except FileNotFoundError:
                continue
            except Exception as e:
                log.error("tester.run.error", runner=name, error=str(e))
                continue
        return "No test runner found."


__all__ = ["TesterAgent"]
