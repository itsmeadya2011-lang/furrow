from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_PROMPT
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings

logger = get_logger("tester")


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        logger.debug("testing_started", goal=goal, tasks=len(tasks))
        test_output = ""
        try:
            test_output = await self._run_tests()
        except Exception as e:
            logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        try:
            result = await self._evaluate(goal, test_output)
        except Exception as e:
            logger.error("test_evaluate_failed", error=str(e))
            return TestResult(passed=False, summary=f"Tester evaluation error: {e}", failures=[str(e)])

        if result.passed or not result.failures:
            logger.info("testing_complete", passed=True, summary=result.summary)
            return result

        try:
            fix_result = await self._attempt_fix(goal, tasks, result)
        except Exception as e:
            logger.error("fix_attempt_failed", error=str(e))
            return TestResult(
                passed=False,
                summary=f"{result.summary} | Fix attempt failed: {e}",
                failures=result.failures + [f"Fix worker raised: {e}"],
            )

        logger.info("testing_complete", passed=fix_result.passed, summary=fix_result.summary)
        return fix_result

    async def _evaluate(self, goal: str, test_output: str) -> TestResult:
        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)
        try:
            data = json.loads(response)
            return TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            return TestResult(passed="passed" in response.lower(), summary=response, failures=[])

    async def _attempt_fix(
        self, goal: str, tasks: list[TaskModel], result: TestResult
    ) -> TestResult:
        relevant_files = sorted({f for t in tasks for f in t.files})
        failure_details = "\n".join(f"- {f}" for f in result.failures)
        description = (
            f"Fix failing tests for goal: {goal}\n\n"
            f"Test summary: {result.summary}\n\n"
            f"Failures:\n{failure_details}"
        )
        fix_task = TaskModel(
            id="test-fix",
            description=description,
            files=relevant_files,
        )

        try:
            worker = WorkerAgent(task=fix_task, client=self.client)
            worker_output = await worker.run()
        except Exception as e:
            return TestResult(
                passed=False,
                summary=f"{result.summary} | Fix worker failed: {e}",
                failures=result.failures,
            )

        try:
            re_output = await self._run_tests()
        except Exception as e:
            return TestResult(
                passed=False,
                summary=f"{result.summary} | Re-run error: {e}",
                failures=result.failures + [f"Re-run raised: {e}"],
            )

        re_result = await self._safe_evaluate(goal, re_output)

        if re_result.passed:
            return TestResult(
                passed=True,
                summary=f"{re_result.summary} (fixed by worker: {worker_output[:200]})",
                failures=[],
            )

        return TestResult(
            passed=False,
            summary=f"{result.summary} | After fix attempt: {re_result.summary}",
            failures=re_result.failures or result.failures,
        )

    async def _safe_evaluate(self, goal: str, test_output: str) -> TestResult:
        try:
            return await self._evaluate(goal, test_output)
        except Exception as e:
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

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
            except (FileNotFoundError, Exception):
                continue
        return "No test runner found."