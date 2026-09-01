from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, SessionState, TestResult
from furrow.core.session import (
    SessionCorruptedError,
    SessionManager,
    SessionNotFoundError,
)
from furrow.llm import LLMClient

logger = logging.getLogger(__name__)

console = Console()

# Type alias for event callback
EventCallback = Callable[[str, dict[str, Any]], None]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        session_id: str | None = None,
        session_manager: SessionManager | None = None,
        auto_save: bool = True,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.current_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.current_plan: Plan | None = None
        self.cycles = 0
        self.workspace = self.client.settings.workspace

        self.session_id = session_id
        self.session_manager = session_manager or SessionManager(self.workspace)
        self.auto_save = auto_save
        self._status: str = "running"
        self.on_event = on_event

    @classmethod
    def from_session(
        cls,
        session_id: str,
        client: LLMClient | None = None,
        on_event: EventCallback | None = None,
    ) -> "Orchestrator":
        """Resume an orchestrator from a previously saved session.

        Raises ``SessionNotFoundError`` if the session does not exist and
        ``SessionCorruptedError`` if the session file cannot be parsed.
        """
        # We need the workspace to construct a SessionManager. Use a default
        # workspace first, then override once we know the real one from the
        # loaded state.
        if client is None:
            from furrow.config import settings

            bootstrap_manager = SessionManager(settings.workspace)
        else:
            bootstrap_manager = SessionManager(client.settings.workspace)

        state = bootstrap_manager.load(session_id)

        orch = cls(
            goal=state.goal,
            client=client,
            session_id=state.session_id,
            session_manager=bootstrap_manager,
            on_event=on_event,
        )
        orch.current_goal = state.current_goal
        orch.cycles = state.cycles
        orch._status = state.status
        if state.workspace:
            orch.workspace = Path(state.workspace)
            orch.session_manager = SessionManager(orch.workspace)
        if state.current_plan is not None:
            try:
                orch.current_plan = Plan.model_validate(state.current_plan)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Could not restore current_plan for session %s: %s",
                    session_id,
                    exc,
                )
                orch.current_plan = None
        # Mark the session as running again now that we've resumed it.
        orch._set_status("running")
        orch._persist()
        return orch

    def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event to the registered callback, if any."""
        if self.on_event is not None:
            try:
                self.on_event(event_type, data)
            except Exception as exc:
                logger.warning("Event callback error: %s", exc)

    def _set_status(self, status: str) -> None:
        self._status = status
        if self.auto_save and self.session_id is not None:
            self._persist()

    def _build_state(self) -> SessionState:
        plan_data: Optional[dict[str, Any]] = None
        if self.current_plan is not None:
            try:
                plan_data = self.current_plan.model_dump()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to serialize current_plan: %s", exc)
                plan_data = None
        return SessionState(
            session_id=self.session_id or "",
            goal=self.goal,
            current_goal=self.current_goal,
            cycles=self.cycles,
            current_plan=plan_data,
            status=self._status,  # type: ignore[arg-type]
            workspace=str(self.workspace),
        )

    def _persist(self) -> None:
        """Save current orchestrator state to disk, if a session is active."""
        if not self.session_id:
            return
        try:
            state = self._build_state()
            self.session_manager.save(self.session_id, state)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to persist session %s: %s", self.session_id, exc)

    async def run(self) -> None:
        self._emit_event("status", {
            "status": "running",
            "session_id": self.session_id,
            "goal": self.goal,
            "message": "Orchestrator started",
        })
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\n"
                f"Goal: {self.goal}\n"
                + (
                    f"Session: [cyan]{self.session_id}[/cyan]"
                    if self.session_id
                    else "Session: [dim]none[/dim]"
                ),
                title="Furrow",
            )
        )
        try:
            while True:
                self.cycles += 1
                self._emit_event("cycle_start", {
                    "cycle": self.cycles,
                    "message": f"Starting cycle {self.cycles}",
                })
                console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
                await self._cycle()
                self._emit_event("cycle_complete", {
                    "cycle": self.cycles,
                    "tasks": self._tasks_to_dict(),
                    "message": f"Cycle {self.cycles} completed",
                })
                if self._is_done():
                    self._set_status("completed")
                    self._emit_event("status", {
                        "status": "completed",
                        "session_id": self.session_id,
                        "cycles": self.cycles,
                        "message": "Goal complete. Halting.",
                    })
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    break
                # Persist state after every successful cycle so we can resume.
                self._persist()
        except KeyboardInterrupt:
            self._set_status("paused")
            self._emit_event("status", {
                "status": "paused",
                "session_id": self.session_id,
                "message": "Furrow paused by user",
            })
            console.print("\n[yellow]Furrow paused. Session state saved.[/yellow]")
            raise
        except Exception as exc:
            self._set_status("paused")
            self._persist()
            self._emit_event("error", {
                "error": str(exc),
                "message": f"Error: {exc}",
            })
            raise

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            self._emit_event("log", {
                "message": "Planning phase started",
                "phase": "planning",
            })
            plan = await self.planner.plan(self.current_goal)
        self.current_plan = plan
        self._emit_event("plan_created", {
            "tasks": [{"id": t.id, "description": t.description} for t in plan.tasks],
            "rationale": plan.rationale,
        })
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._emit_event("log", {
                "message": "No tasks planned. Goal may be complete.",
                "phase": "planning",
            })
            return

        # Mark all tasks as running
        for task in plan.tasks:
            task.status = "running"
        self._emit_event("task_update", {
            "tasks": self._tasks_to_dict(),
            "message": f"Starting {len(plan.tasks)} tasks",
        })

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, workspace=self.workspace).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit_event("task_update", {
                    "task_id": task.id,
                    "status": "failed",
                    "error": str(result),
                    "tasks": self._tasks_to_dict(),
                })
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit_event("task_update", {
                    "task_id": task.id,
                    "status": "completed",
                    "tasks": self._tasks_to_dict(),
                })
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            self._emit_event("log", {
                "message": "Testing phase started",
                "phase": "testing",
            })
            test_result = await TesterAgent(client=self.client).run(self.current_goal, plan.tasks)

        if test_result.passed:
            self._emit_event("log", {
                "message": f"Tests passed: {test_result.summary}",
                "phase": "testing",
                "passed": True,
            })
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit_event("log", {
                "message": f"Tests failed: {test_result.summary}",
                "phase": "testing",
                "passed": False,
                "failures": test_result.failures,
            })
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.current_goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
            console.print(f"[yellow]Reached max_cycles ({self.client.settings.max_cycles}). Halting.[/yellow]")
            return True
        completed = sum(1 for t in self._get_tasks() if t.status == "completed")
        failed = sum(1 for t in self._get_tasks() if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._get_tasks()):
            return True
        return False

    def _get_tasks(self) -> list[Any]:
        if self.current_plan is not None:
            return self.current_plan.tasks
        return []

    def _tasks_to_dict(self) -> list[dict[str, Any]]:
        """Convert current tasks to a list of dictionaries for serialization."""
        tasks = self._get_tasks()
        return [
            {
                "id": t.id,
                "description": t.description,
                "status": t.status,
                "result": t.result,
            }
            for t in tasks
        ]
