from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

import structlog
from rich.console import Console

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.config import settings as global_settings
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state_file: Path | None = None,
        on_update: Callable[[str], None] | Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        effective_settings = settings or global_settings
        self.client = client or LLMClient(settings=effective_settings)
        self.planner = PlannerAgent(client=self.client, settings=effective_settings)
        self.tester = TesterAgent(client=self.client, settings=effective_settings)
        self.settings = effective_settings
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.completed_tasks: set[str] = set()
        self.failed_tasks: dict[str, str] = {}
        self.state_file = state_file or Path.cwd() / ".furrow_state.json"
        self.on_update = on_update or (lambda msg: None)
        self._load_state()

    async def _emit(self, msg: str) -> None:
        """Emit update to both console and callback."""
        console.print(msg)
        result = self.on_update(msg)
        if asyncio.iscoroutine(result):
            await result

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = data.get("cycles", 0)
                self.all_tasks = [TaskModel(**t) for t in data.get("tasks", [])]
                self.completed_tasks = set(data.get("completed", []))
                self.failed_tasks = data.get("failed", {})
                self.goal = data.get("goal", self.goal)
                logger.info("state_loaded", cycles=self.cycles, tasks=len(self.all_tasks))
            except Exception as e:
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        data = {
            "cycles": self.cycles,
            "goal": self.goal,
            "tasks": [t.model_dump() for t in self.all_tasks],
            "completed": list(self.completed_tasks),
            "failed": self.failed_tasks,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        await self._emit(f"CYCLE:═══ Furrow Started ═══\\nGoal: {self.goal}")
        max_cycles = self.settings.max_cycles or float("inf")

        while self.cycles < max_cycles:
            self.cycles += 1
            await self._emit(f"CYCLE:═══ Cycle {self.cycles} ═══")
            await self._cycle()
            self._save_state()

            if self._is_done():
                await self._emit("DONE:Goal complete. Halting.")
                self.state_file.unlink(missing_ok=True)
                break
        else:
            await self._emit(f"DONE:Max cycles ({max_cycles}) reached. State saved to {self.state_file}")

    def _get_pending_tasks(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Filter tasks whose dependencies are all completed."""
        pending = []
        for task in tasks:
            if task.id in self.completed_tasks:
                continue
            if task.id in self.failed_tasks:
                continue
            deps_met = all(dep in self.completed_tasks for dep in task.dependencies)
            if deps_met:
                pending.append(task)
        return pending

    async def _cycle(self) -> None:
        # Get pending tasks from previous plan or create new plan
        pending = self._get_pending_tasks(self.all_tasks)

        if not pending:
            plan = await self.planner.plan(self.goal, self.all_tasks)
            await self._emit(f"PLAN:{plan.rationale}")

            if not plan.tasks:
                await self._emit("No tasks planned. Goal may be complete.")
                return

            # Add new tasks to our task list
            for task in plan.tasks:
                if task.id not in {t.id for t in self.all_tasks}:
                    self.all_tasks.append(task)
            pending = self._get_pending_tasks(self.all_tasks)

        if not pending:
            await self._emit("All tasks blocked on dependencies or completed.")
            return

        # Execute pending tasks in parallel (respecting max_parallel_tasks)
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

        async def run_task(task: TaskModel):
            async with semaphore:
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                return await worker.run()

        tasks_coro = [run_task(task) for task in pending]
        results = await asyncio.gather(*tasks_coro, return_exceptions=True)

        for task, result in zip(pending, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self.failed_tasks[task.id] = str(result)
                await self._emit(f"TASK_FAIL:Task {task.id} failed: {result}")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = str(result)
                self.completed_tasks.add(task.id)
                await self._emit(f"TASK_DONE:Task {task.id} completed: {result}")
                logger.info("task_completed", task_id=task.id)

        test_result = await self.tester.run(self.goal, self.all_tasks)

        if test_result.passed:
            await self._emit(f"TEST_PASS:Tests passed: {test_result.summary}")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            await self._emit(f"TEST_FAIL:Tests failed: {test_result.summary}")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("Will attempt fix in next cycle.")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)
            logger.warning("tests_failed", failures=test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False

        total = len(self.all_tasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)

        if failed > 0:
            return False

        return completed >= total
