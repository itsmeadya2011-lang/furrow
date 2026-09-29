from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import (
    Plan,
    Settings,
    State,
    TaskModel,
    TestResult,
    load_state,
    save_state,
)
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    """Runs the plan -> execute -> test loop until the goal is complete."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        max_cycles: int | None = None,
    ) -> None:
        self.client = client or LLMClient(settings=settings)
        self.settings = self.client.settings
        self.goal = goal
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_plan: Plan | None = None
        self.last_result: TestResult | None = None
        self.state: State | None = None
        self.resumed = False
        self.max_cycles = max_cycles if max_cycles is not None else self.settings.max_cycles
        self._resume()

    # ------------------------------------------------------------------ #
    # Persistent state
    # ------------------------------------------------------------------ #
    def _resume(self) -> None:
        """Load an interrupted session so the user can resume the same goal."""
        try:
            state = load_state(self.settings.workspace)
        except Exception:
            return
        if not state.goal or state.goal != self.goal:
            return
        # A state whose tasks all completed is considered finished -> start fresh.
        finished = bool(state.tasks) and all(t.status == "completed" for t in state.tasks)
        if finished:
            return
        self.goal = state.goal
        self.cycles = state.cycles
        if state.max_cycles:
            self.max_cycles = state.max_cycles
        self.tasks = list(state.tasks)
        self.state = state
        self.resumed = True

    def _persist(self) -> None:
        state = State(
            goal=self.goal,
            cycles=self.cycles,
            max_cycles=self.max_cycles,
            tasks=self.tasks,
            failures=(
                self.last_result.failures
                if self.last_result and not self.last_result.passed
                else []
            ),
        )
        self.state = state
        try:
            save_state(state, self.settings.workspace)
        except Exception as e:  # never let persistence break the loop
            console.print(f"[yellow]Warning: could not persist state: {e}[/yellow]")

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #
    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        if self.resumed:
            console.print(
                f"[yellow]Resuming goal from cycle {self.cycles}: {self.goal}[/yellow]"
            )
        async for update in self.run_stream():
            self._print_update(update)

    async def run_stream(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                yield {"type": "done", "reason": "max_cycles"}
                break

            self.cycles += 1
            yield {"type": "cycle_start", "cycle": self.cycles, "goal": self.goal}

            plan = await self.planner.plan(self.goal)
            self.last_plan = plan
            self.tasks = plan.tasks
            yield {"type": "plan", "plan": plan.model_dump()}

            if not plan.tasks:
                yield {"type": "done", "reason": "no_tasks"}
                break

            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                worker_tasks = [
                    WorkerAgent(task=task, client=self.client).run() for task in plan.tasks
                ]
                results = await asyncio.gather(*worker_tasks, return_exceptions=True)

            for task, result in zip(plan.tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                else:
                    task.status = "completed"
                    task.result = result

            yield {"type": "task_result", "results": [t.model_dump() for t in plan.tasks]}

            with Status("[bold yellow]Testing...", console=console):
                test_result = await self.tester.run(self.goal, plan.tasks)
            self.last_result = test_result
            yield {"type": "test", "test_result": test_result.model_dump()}

            if not test_result.passed:
                self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

            self._persist()

            if test_result.passed:
                yield {"type": "done", "reason": "tests_passed"}
                break

    def _print_update(self, update: dict[str, Any]) -> None:
        utype = update.get("type")
        if utype == "cycle_start":
            console.print(f"\n[bold cyan]═══ Cycle {update['cycle']} ═══[/bold cyan]")
            console.print(f"[cyan]Goal:[/cyan] {self.goal}")
        elif utype == "plan":
            console.print(Panel(Pretty(update["plan"]), title="Plan", border_style="blue"))
        elif utype == "task_result":
            for r in update["results"]:
                if r["status"] == "completed":
                    console.print(f"[green]Task {r['id']} completed[/green]")
                else:
                    console.print(f"[red]Task {r['id']} failed: {r['result']}[/red]")
        elif utype == "test":
            tr = update["test_result"]
            if tr["passed"]:
                console.print(f"[green]Tests passed: {tr['summary']}[/green]")
            else:
                console.print(f"[red]Tests failed: {tr['summary']}[/red]")
                for f in tr.get("failures", []):
                    console.print(f"  • {f}")
                console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        elif utype == "done":
            reason = update["reason"]
            if reason == "tests_passed":
                console.print("[bold green]Goal complete. Halting.[/bold green]")
            elif reason == "max_cycles":
                console.print("[yellow]Max cycles reached. Halting.[/yellow]")
            else:
                console.print("[yellow]No tasks planned. Halting.[/yellow]")
