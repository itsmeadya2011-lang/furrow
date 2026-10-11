from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from furrow.agents.prompts import TESTER_FIX_PROMPT, TESTER_PROMPT
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import get_logger

if TYPE_CHECKING:
    from furrow.config import Settings


class TesterAgent:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = settings or self.client.settings
        self.logger = get_logger("tester")

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        test_output = ""
        try:
            test_output = await self._run_tests()
        except (asyncio.CancelledError, OSError) as e:
            self.logger.error("test_run_failed", error=str(e))
            return TestResult(passed=False, summary=str(e), failures=[str(e)])

        prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output:\n{test_output}\n"
        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        try:
            data = json.loads(response)
            result = TestResult(**data)
        except (json.JSONDecodeError, ValueError):
            result = TestResult(passed="passed" in response.lower(), summary=response, failures=[])

        self.logger.info(
            "test_result",
            passed=result.passed,
            summary=result.summary,
            failure_count=len(result.failures),
        )

        # If tests failed, try to fix them
        if not result.passed and result.failures:
            result = await self._try_fix_failures(goal, tasks, test_output, result)

        return result

    async def _try_fix_failures(
        self, goal: str, tasks: list[TaskModel], test_output: str, initial_result: TestResult
    ) -> TestResult:
        """Attempt to fix failing tests by creating fix tasks and running workers."""
        from furrow.agents.worker import WorkerAgent

        # Build context about what was done
        task_context = "\n".join(
            f"- {t.id}: {t.description} [{t.status}]"
            + (f" -> {t.result[:100]}" if t.result else "")
            for t in tasks
        )

        prompt = (
            f"{TESTER_FIX_PROMPT}\n\nGoal: {goal}\n\nRecent tasks:\n{task_context}\n\nTest output:\n{test_output}\n\nFailures:\n"
            + "\n".join(f"- {f}" for f in initial_result.failures)
        )

        response = await self.client.complete(prompt, model=self.client.settings.tester_model)

        try:
            fix_plan = json.loads(response)
            fix_tasks = fix_plan.get("fix_tasks", [])
        except (json.JSONDecodeError, ValueError):
            # Fallback: create generic fix tasks from failures
            fix_tasks = [
                {"description": f"Fix: {failure}", "files": []}
                for failure in initial_result.failures[:3]
            ]

        if not fix_tasks:
            return initial_result

        self.logger.info("attempting_fixes", fix_task_count=len(fix_tasks))

        # Execute fix tasks
        for fix_task in fix_tasks:
            task_model = TaskModel(
                id=f"fix_{len(tasks) + 1}",
                description=fix_task["description"],
                files=fix_task.get("files", []),
                dependencies=[],
            )
            worker = WorkerAgent(task=task_model, client=self.client, settings=self.settings)
            try:
                await worker.run()
            except (asyncio.CancelledError, OSError, RuntimeError) as e:
                self.logger.error("fix_task_failed", error=str(e))
                return TestResult(passed=False, summary=f"Fix failed: {e}", failures=[str(e)])

        # Re-run tests
        try:
            new_output = await self._run_tests()
            prompt = f"{TESTER_PROMPT}\n\nGoal: {goal}\n\nTest output after fixes:\n{new_output}\n"
            response = await self.client.complete(prompt, model=self.client.settings.tester_model)
            data = json.loads(response)
            result = TestResult(**data)
            self.logger.info("retest_result", passed=result.passed, summary=result.summary)
            return result
        except (asyncio.CancelledError, OSError, json.JSONDecodeError, ValueError) as e:
            self.logger.error("retest_failed", error=str(e))
            return TestResult(passed=False, summary=f"Re-test failed: {e}", failures=[str(e)])

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
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.settings.workspace,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                    return stdout.decode() + stderr.decode()
                except asyncio.TimeoutError:
                    proc.kill()
                    continue
            except (FileNotFoundError, OSError):
                continue
        return "No test runner found."
