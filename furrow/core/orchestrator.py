from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        output_callback=None,
    ) -> None:
        from furrow.config import settings as global_settings

        self.goal = goal
        self.client = client or LLMClient(settings=settings or global_settings)
        self.settings = settings or global_settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.output_callback = output_callback

    def _log(self, message: str | Panel) -> None:
        console.print(message)
        if self.output_callback:
            self.output_callback(str(message))

    async def run(self) -> None:
        self._log(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._log("[yellow]Max cycles reached. Halting.[/yellow]")
                break
            self.cycles += 1
            self._log(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._log("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        # Get file context for planner
        file_context = self._get_file_context()

        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, file_context)
        self._log(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._log("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Merge new tasks with existing ones (avoid duplicates by id)
        for new_task in plan.tasks:
            existing = next((t for t in self.all_tasks if t.id == new_task.id), None)
            if not existing:
                self.all_tasks.append(new_task)

        # Execute only pending tasks
        pending_tasks = [t for t in self.all_tasks if t.status == "pending"]
        if pending_tasks:
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

                async def run_with_semaphore(task: TaskModel):
                    async with semaphore:
                        return await WorkerAgent(
                            task=task, client=self.client, settings=self.settings
                        ).run()

                tasks = [run_with_semaphore(task) for task in pending_tasks]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for task, result in zip(pending_tasks, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self._log(f"[red]Task {task.id} failed: {result}[/red]")
                elif isinstance(result, str):
                    task.status = "completed"
                    task.result = result
                    self._log(f"[green]Task {task.id} completed[/green]")
                else:
                    task.status = "completed"
                    task.result = str(result) if result is not None else None
                    self._log(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, self.all_tasks
            )

        if test_result.passed:
            self._log(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._log(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._log(f"  • {failure}")
            self._log("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_file_context(self) -> str:
        """Get a summary of the workspace files for the planner."""
        files = self.client.list_files(self.settings.workspace)
        if not files:
            return "Workspace is empty."
        # Limit to avoid token overflow
        limited = files[:50]
        return (
            "Files in workspace:\n"
            + "\n".join(f"  {f}" for f in limited)
            + (f"\n  ... and {len(files) - 50} more" if len(files) > 50 else "")
        )

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
