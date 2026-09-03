from __future__ import annotations

import asyncio
import shutil
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, settings
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._history: list[Plan] = []
        self._semaphore = asyncio.Semaphore(max(1, settings.max_parallel_tasks))

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        max_cycles = settings.max_cycles or None
        consecutive_empty = 0
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            planned_count = await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break
            if planned_count == 0:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    console.print(
                        "[yellow]Planner returned no tasks twice in a row. Halting.[/yellow]"
                    )
                    break
            else:
                consecutive_empty = 0
            if max_cycles is not None and self.cycles >= max_cycles:
                console.print(
                    f"[yellow]Reached max_cycles={max_cycles}. Halting.[/yellow]"
                )
                break

    async def _cycle(self) -> int:
        """Run one planning + execution + test cycle. Returns the number of tasks planned."""
        snapshot = _snapshot_workspace()

        with Status("[bold yellow]Planning...", console=console):
            try:
                plan = await self.planner.plan(self.goal)
            except Exception as e:
                console.print(f"[red]Planning failed: {e}[/red]")
                _restore_workspace(snapshot)
                return 0

        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return 0

        self._history.append(plan)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                self._run_task(task) for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            return len(plan.tasks)

        console.print(f"[red]Tests failed: {test_result.summary}[/red]")
        for failure in test_result.failures:
            console.print(f"  • {failure}")

        # If tests failed and nothing was actually changed, restore the snapshot
        # so the next cycle starts clean. Otherwise keep changes for the fixer
        # to iterate on.
        if not any(t.status == "completed" for t in plan.tasks):
            _restore_workspace(snapshot)
            console.print("[yellow]Restored workspace after test failure.[/yellow]")
        else:
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            failures = "\n".join(test_result.failures) or test_result.summary
            self.goal = f"Original goal: {self.goal}\n\nFix failing tests:\n{failures}"
        return len(plan.tasks)

    async def _run_task(self, task: TaskModel) -> str:
        async with self._semaphore:
            return await WorkerAgent(task=task, client=self.client).run()

    def _is_done(self) -> bool:
        if not self._history:
            return False
        last_plan = self._history[-1]
        completed = sum(1 for t in last_plan.tasks if t.status == "completed")
        failed = sum(1 for t in last_plan.tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(last_plan.tasks):
            return True
        return False


def _snapshot_workspace() -> str | None:
    """Return current HEAD sha, or None if no git repo / git unavailable."""
    if not shutil.which("git"):
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=settings.workspace,
        )
        if result.returncode != 0:
            return None
        sha = result.stdout.strip()
        # Make sure there are no uncommitted changes we would lose.
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=settings.workspace,
        )
        if status.returncode == 0 and status.stdout.strip():
            subprocess.run(
                ["git", "stash", "push", "-u", "-m", "furrow-snapshot"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=settings.workspace,
            )
        return sha
    except Exception:
        return None


def _restore_workspace(sha: str | None) -> None:
    if not sha or not shutil.which("git"):
        return
    try:
        subprocess.run(
            ["git", "reset", "--hard", sha],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=settings.workspace,
        )
        # Pop any snapshot stash we created.
        subprocess.run(
            ["git", "stash", "pop"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=settings.workspace,
        )
    except Exception:
        pass