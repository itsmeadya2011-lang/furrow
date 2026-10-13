from __future__ import annotations

import asyncio

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.events import EventBus
from furrow.llm import LLMClient


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.original_goal = goal
        self.goal = goal
        self.client = client or LLMClient()
        self.event_bus = event_bus or EventBus()
        self.cycles = 0
        self.history: list[dict] = []
        self.last_test_result: TestResult | None = None
        self.plan: Plan | None = None

    async def run(self) -> None:
        self.event_bus.emit({"type": "start", "goal": self.goal})
        while True:
            self.cycles += 1
            self.event_bus.emit(
                {"type": "cycle.start", "cycle": self.cycles, "goal": self.goal}
            )

            try:
                plan = await PlannerAgent(client=self.client).plan(
                    self.goal, history=self.history
                )
            except ValueError as e:
                self.event_bus.emit(
                    {
                        "type": "error",
                        "stage": "planner",
                        "error": str(e),
                    }
                )
                self._halt("planner_error")
                return

            self.plan = plan
            if plan.tasks is None:
                plan.tasks = []
            self.event_bus.emit(
                {
                    "type": "plan",
                    "cycle": self.cycles,
                    "plan": plan.model_dump(),
                }
            )

            if plan is None or not plan.tasks:
                self.event_bus.emit(
                    {
                        "type": "cycle.end",
                        "cycle": self.cycles,
                    }
                )
                self._halt("no_tasks")
                return

            await self._execute_tasks(plan.tasks, self.cycles)

            self.event_bus.emit({"type": "test.start", "cycle": self.cycles})

            test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )
            self.last_test_result = test_result
            self.event_bus.emit(
                {
                    "type": "test.result",
                    "cycle": self.cycles,
                    "result": test_result.model_dump(),
                }
            )

            self.history.append(
                {
                    "cycle": self.cycles,
                    "goal": self.goal,
                    "plan": plan.model_dump(),
                    "test_result": test_result.model_dump(),
                }
            )

            self.event_bus.emit({"type": "cycle.end", "cycle": self.cycles})

            reason = self._should_halt(test_result)
            if reason is not None:
                self.event_bus.emit(
                    {
                        "type": "halt",
                        "reason": reason,
                        "cycles": self.cycles,
                    }
                )
                break

            if not test_result.passed:
                self.goal = (
                    f"{self.original_goal}\n\nFix these failing tests:\n"
                    + "\n".join(test_result.failures)
                )

    async def _execute_tasks(self, tasks: list[TaskModel], cycle: int) -> None:
        sem = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _run_one(task: TaskModel) -> None:
            async with sem:
                self.event_bus.emit(
                    {
                        "type": "task.start",
                        "cycle": cycle,
                        "task_id": task.id,
                    }
                )
                try:
                    summary = await WorkerAgent(
                        task=task, client=self.client
                    ).run()
                    task.status = "completed"
                    task.result = summary
                    self.event_bus.emit(
                        {
                            "type": "task.complete",
                            "cycle": cycle,
                            "task_id": task.id,
                            "summary": summary,
                        }
                    )
                except Exception:  # pragma: no cover  # noqa: BLE001
                    task.status = "failed"
                    task.result = "Worker execution failed"
                    self.event_bus.emit(
                        {
                            "type": "task.failed",
                            "cycle": cycle,
                            "task_id": task.id,
                            "error": "Worker execution failed",
                        }
                    )

        coros = [_run_one(task) for task in tasks]
        await asyncio.gather(*coros)

    def _should_halt(self, test_result: TestResult) -> str | None:
        max_cycles = self.client.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            return "max_cycles"
        plan = self.plan
        if plan is None:
            return "no_tasks"
        if test_result.passed and all(
            t.status == "completed" for t in plan.tasks
        ):
            return "complete"
        return None

    def _halt(self, reason: str) -> None:
        self.event_bus.emit(
            {"type": "halt", "reason": reason, "cycles": self.cycles}
        )