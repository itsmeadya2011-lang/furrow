from __future__ import annotations

import asyncio
import io
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from tenacity import retry, stop_after_attempt, wait_exponential

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

console = Console()

# Strip rich console markup (e.g. [bold green], [/bold]) for plain-text output.
_RICH_TAG = re.compile(r"\[/?[a-zA-Z0-9 _\/.:-]*\]")


def _strip_rich(text: str) -> str:
    return _RICH_TAG.sub("", text)


def _render(renderable: Any) -> str:
    """Render a rich renderable to a plain-text string."""
    buf = io.StringIO()
    cap = Console(file=buf, width=100)
    cap.print(renderable)
    return buf.getvalue().rstrip("\n")


OutputCallback = Callable[[str], Any]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_output: OutputCallback | None = None,
        state_file: Path | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings = settings or default_settings
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.on_output = on_output
        self.state_file = state_file or (self.settings.workspace / ".furrow_state.json")
        self._load_state()

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _emit(self, message: str) -> None:
        """Print to console and forward plain text to the output callback."""
        console.print(message)
        if self.on_output:
            outcome = self.on_output(_strip_rich(message))
            if asyncio.iscoroutine(outcome):
                asyncio.ensure_future(outcome)

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _state_payload(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
        }

    def save_state(self) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(self._state_payload(), indent=2))
        except OSError:
            # State persistence is best-effort; never crash the loop over it.
            pass

    def _load_state(self) -> None:
        try:
            if not self.state_file.exists():
                return
            data = json.loads(self.state_file.read_text())
            self.goal = data.get("goal", self.goal)
            self.cycles = data.get("cycles", 0)
            self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
        except (OSError, json.JSONDecodeError, ValueError):
            # Corrupt or missing state is not fatal; start fresh.
            self.tasks = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        self.client.validate()
        self._emit(
            _render(
                Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
            )
        )
        while True:
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._emit(
                    f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Halting.[/yellow]"
                )
                break
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self.save_state()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self._retry(self.planner.plan, self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Merge new tasks into the persistent task list (by id).
        existing_ids = {t.id for t in self.tasks}
        for task in plan.tasks:
            if task.id not in existing_ids:
                self.tasks.append(task)

        pending = [t for t in self.tasks if t.status == "pending"]
        if not pending:
            self._emit("[yellow]No pending tasks. Goal may be complete.[/yellow]")
            return

        # Execute tasks whose dependencies are satisfied, in parallel waves.
        while pending:
            ready = [
                t
                for t in pending
                if all(
                    any(
                        existing.id == dep and existing.status == "completed"
                        for existing in self.tasks
                    )
                    for dep in t.dependencies
                )
            ]
            if not ready:
                # Remaining pending tasks have unresolvable dependencies.
                self._emit("[red]Pending tasks have unsatisfied dependencies; skipping.[/red]")
                break

            workers = [WorkerAgent(task=task, client=self.client) for task in ready]
            with Status("[bold yellow]Executing tasks in parallel...", console=console):
                results = await asyncio.gather(
                    *[self._retry(worker.run) for worker in workers],
                    return_exceptions=True,
                )

            for task, result in zip(ready, results, strict=False):
                if isinstance(result, Exception):
                    task.status = "failed"
                    task.result = str(result)
                    self._emit(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = cast(str, result)
                    self._emit(f"[green]Task {task.id} completed[/green]")

            pending = [t for t in self.tasks if t.status == "pending"]

        with Status("[bold yellow]Testing...", console=console):
            test_result = await self._retry(
                TesterAgent(client=self.client, settings=self.settings).run,
                self.goal,
                self.tasks,
            )

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.tasks:
            return True
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _retry(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)
