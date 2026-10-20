from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()
log = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.on_event: Callable[[str], Awaitable[None]] | None = on_event
        self.cycles = 0
        self.plan: Plan | None = None
        self.focus: str | None = None
        self._stalls: int = 0
        self._last_signature: tuple[tuple[str, ...], str] | None = None

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        while True:
            self.cycles += 1
            await self._emit(f"cycle {self.cycles} start")
            log.info("cycle start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            keep_going = await self._cycle()
            if not keep_going:
                break
            if self._is_done():
                await self._emit("goal complete")
                log.info("goal complete", cycle=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                await self._emit(f"stopping after {self.cycles} cycle(s)")
                console.print(f"[yellow]Stopping after {self.cycles} cycle(s).[/yellow]")
                break

    async def _cycle(self) -> bool:
        """Run a single plan / execute / test cycle.

        Returns False when the run loop should stop (planner failure or stall).
        """
        plan = await self._plan_with_retries(self._planner_goal())
        if plan is None:
            return False
        self.plan = plan
        await self._emit(f"plan ready: {len(plan.tasks)} task(s)")
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self.focus = None
            self._stalls = 0
            self._last_signature = None
            return True

        await self._run_tasks(plan.tasks)
        completed_count = sum(1 for task in plan.tasks if task.status == "completed")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, self.plan.tasks)

        failed_ids = sorted(t.id for t in plan.tasks if t.status == "failed")
        test_failures_text = "" if test_result.passed else "\n".join(test_result.failures)
        signature = (tuple(failed_ids), test_failures_text)

        if completed_count > 0:
            self._stalls = 0
        elif signature == self._last_signature:
            self._stalls += 1
        else:
            self._stalls = 0
        self._last_signature = signature

        if test_result.passed:
            await self._emit(f"tests passed: {test_result.summary}")
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            self.focus = None
            if self._stalls >= 3:
                await self._emit(
                    f"stalled: no progress after {self._stalls} consecutive failed cycles"
                )
                console.print(
                    f"[yellow]Stalled: no progress after {self._stalls} consecutive failed "
                    f"cycle(s); stopping.[/yellow]"
                )
                return False
            return True

        await self._emit(f"tests failed: {test_result.summary}")
        console.print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            console.print(f"  • {failure}")

        self.focus = test_failures_text

        if self._stalls >= 3:
            await self._emit(f"stalled: no progress after {self._stalls} consecutive failed cycles")
            console.print(
                f"[yellow]Stalled: no progress after {self._stalls} consecutive failed "
                f"cycle(s); stopping.[/yellow]"
            )
            return False

        console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        return True

    def _planner_goal(self) -> str:
        if self.focus:
            return f"{self.goal}\n\nAdditional focus for this cycle:\n{self.focus}"
        return self.goal

    async def _plan_with_retries(self, goal: str) -> Plan | None:
        backoff = (1.0, 2.0, 4.0)
        for attempt in range(1, 4):
            try:
                return await self.planner.plan(goal)
            except Exception as exc:  # noqa: BLE001 - planner resilience: catch any planner error
                error = str(exc)
                log.error("planner failed", attempt=attempt, error=error)
                await self._emit(f"planner failed: {error}")
                await asyncio.sleep(backoff[attempt - 1])
        console.print("[red]Planner failed after 3 attempts; stopping the run.[/red]")
        return None

    async def _run_tasks(self, tasks: list[TaskModel]) -> None:
        """Run tasks honouring dependencies and the max-parallel cap."""
        if not tasks:
            return
        max_parallel = max(1, self.settings.max_parallel_tasks)
        semaphore = asyncio.Semaphore(max_parallel)
        known = {task.id for task in tasks}
        cyclic = self._cyclic_task_ids(tasks)
        events = {task.id: asyncio.Event() for task in tasks}

        async def run_one(task: TaskModel) -> None:
            for dep in task.dependencies:
                if dep in known and dep not in cyclic:
                    await events[dep].wait()
            async with semaphore:
                await self._emit(f"task {task.id} start")
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                except Exception as exc:  # noqa: BLE001 - a failing worker must not kill the loop
                    task.status = "failed"
                    task.result = str(exc)
                    log.error("task failed", task_id=task.id, error=str(exc))
                    await self._emit(f"task {task.id} failed: {exc}")
                else:
                    task.status = "completed"
                    task.result = result
                    await self._emit(f"task {task.id} done")
            events[task.id].set()

        await asyncio.gather(*(run_one(task) for task in tasks))

    def _cyclic_task_ids(self, tasks: list[TaskModel]) -> set[str]:
        """Return task ids that take part in a dependency cycle (known ids only)."""
        known = {task.id for task in tasks}
        adjacency = {task.id: [dep for dep in task.dependencies if dep in known] for task in tasks}
        color: dict[str, int] = {task_id: 0 for task_id in known}
        stack: list[str] = []
        cyclic: set[str] = set()

        def dfs(node: str) -> None:
            color[node] = 1
            stack.append(node)
            for nxt in adjacency[node]:
                if color[nxt] == 1:
                    cyclic.update(stack[stack.index(nxt) :])
                elif color[nxt] == 0:
                    dfs(nxt)
            stack.pop()
            color[node] = 2

        for task_id in known:
            if color[task_id] == 0:
                dfs(task_id)
        return cyclic

    async def _emit(self, event: str) -> None:
        if self.on_event is None:
            return
        try:
            await self.on_event(event)
        except Exception:
            log.exception("on_event listener raised", listener_event=event)

    def _get_tasks(self) -> list[TaskModel]:
        if self.plan is None:
            return []
        return self.plan.tasks

    def _is_done(self) -> bool:
        if self.plan is None:
            return False
        tasks = self.plan.tasks
        if not tasks:
            return True
        if any(task.status == "failed" for task in tasks):
            return False
        return all(task.status == "completed" for task in tasks)
