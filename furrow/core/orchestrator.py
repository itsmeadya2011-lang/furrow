from __future__ import annotations

import asyncio
from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import FileEdit, Plan, Settings, TaskModel, WorkerResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()

_MAX_CONTEXT_CHARS = 4000


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings if settings is not None else default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.on_event = on_event
        self.cycles = 0
        self.plan: Plan | None = None
        self.history: list[str] = []
        self._semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        self._emit("start", {"goal": self.goal})

        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit("cycle", {"cycle": self.cycles})

            done = await self._cycle()
            if done:
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self._emit("done", {"cycles": self.cycles})
                break

            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                console.print("[yellow]Cycle limit reached. Halting.[/yellow]")
                self._emit("stopped", {"cycles": self.cycles, "reason": "max_cycles"})
                break

    async def _cycle(self) -> bool:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, context=self._build_context())
        self.plan = plan

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))
        self._emit(
            "plan",
            {"tasks": [t.model_dump() for t in plan.tasks], "rationale": plan.rationale},
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return True

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(*[self._run_task(task) for task in plan.tasks])

        for task, result in results:
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                self._emit(
                    "task_done",
                    {"task_id": task.id, "status": "failed", "summary": str(result)},
                )
                continue

            for edit in result.edits:
                await self._apply_edit(edit)

            task.status = "completed"
            task.result = result.summary
            console.print(f"[green]Task {task.id} completed: {result.summary}[/green]")
            self._emit(
                "task_done",
                {"task_id": task.id, "status": "completed", "summary": result.summary},
            )

        self.history.extend(
            f"Cycle {self.cycles}: task {task.id} ({task.status}): {task.result}"
            for task in plan.tasks
        )

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, plan.tasks, context=self._build_context()
            )

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")

        self._emit(
            "test",
            {
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": list(test_result.failures),
            },
        )

        self.history.append(
            f"Cycle {self.cycles}: tests {'passed' if test_result.passed else 'failed'}: "
            f"{test_result.summary}"
        )
        for failure in test_result.failures:
            self.history.append(f"Cycle {self.cycles}: failure: {failure}")

        return all(t.status == "completed" for t in plan.tasks) and test_result.passed

    async def _run_task(self, task: TaskModel) -> tuple[TaskModel, WorkerResult | Exception]:
        async with self._semaphore:
            self._emit("task_start", {"task_id": task.id})
            try:
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                result = await worker.run()
            except Exception as e:
                return task, e
            return task, result

    async def _apply_edit(self, edit: FileEdit) -> None:
        root = self.settings.workspace.resolve()
        target = (self.settings.workspace / edit.path).resolve()
        if target != root and root not in target.parents:
            console.print(f"[red]Refusing edit outside workspace: {edit.path}[/red]")
            self._emit("error", {"error": f"Refusing edit outside workspace: {edit.path}"})
            return

        await self.client.write_file(target, edit.content)
        console.print(f"[dim]Updated {edit.path}[/dim]")
        self._emit("edit", {"path": edit.path})

    def _build_context(self) -> str:
        if not self.history:
            return ""
        context = "\n".join(self.history)
        if len(context) > _MAX_CONTEXT_CHARS:
            return context[-_MAX_CONTEXT_CHARS:]
        return context

    def _is_done(self) -> bool:
        if self.plan is None or not self.plan.tasks:
            return True
        return all(t.status == "completed" for t in self.plan.tasks)

    def _emit(self, event: str, data: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event, data)
        except Exception:
            pass
