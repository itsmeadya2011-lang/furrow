from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult, settings
from furrow.llm import LLMClient
from furrow import logger

console = Console()
log = logger.bind(component="orchestrator")


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, on_progress: Callable[[dict], Any] | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[Any] = []
        self.state_file = settings.workspace / ".furrow_state.json"
        self.on_progress = on_progress
        self._load_state()
        log.info("orchestrator_initialized", goal=goal, cycles=self.cycles, existing_tasks=len(self.all_tasks))

    def _emit(self, msg: dict) -> None:
        if self.on_progress:
            asyncio.create_task(self.on_progress(msg))
        # Also log to console
        if msg.get("type") == "log":
            console.print(msg.get("message", ""))

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.all_tasks = data.get("tasks", [])
                self.cycles = data.get("cycles", 0)
                if data.get("goal") and not self.goal:
                    self.goal = data["goal"]
            except Exception:
                pass

    def _save_state(self) -> None:
        data = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "files": t.files,
                    "dependencies": t.dependencies,
                    "status": t.status,
                    "result": t.result,
                }
                for t in self.all_tasks
            ],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    async def run(self) -> None:
        self._emit({"type": "log", "message": f"[bold green]Furrow[/bold green]\\nGoal: {self.goal}"})
        log.info("starting_run", goal=self.goal)
        while True:
            if settings.max_cycles > 0 and self.cycles >= settings.max_cycles:
                self._emit({"type": "log", "message": f"[yellow]Reached max cycles ({settings.max_cycles}). Stopping.[/yellow]"})
                log.warning("max_cycles_reached", max_cycles=settings.max_cycles)
                break
            self.cycles += 1
            self._emit({"type": "cycle", "cycle": self.cycles})
            log.info("cycle_start", cycle=self.cycles)
            await self._cycle()
            self._save_state()
            if self._is_done():
                self._emit({"type": "log", "message": "[bold green]Goal complete. Halting.[/bold green]"})
                self._emit({"type": "done"})
                log.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        # Provide context about existing files and completed tasks
        context = self._build_context()
        
        self._emit({"type": "log", "message": "[bold yellow]Planning...[/bold yellow]"})
        log.info("planning", cycle=self.cycles)
        plan = await self.planner.plan(self.goal, context)
        self._emit({"type": "plan", "rationale": plan.rationale, "tasks": [t.model_dump() for t in plan.tasks]})
        log.info("plan_created", cycle=self.cycles, num_tasks=len(plan.tasks), rationale=plan.rationale)

        if not plan.tasks:
            self._emit({"type": "log", "message": "[yellow]No tasks planned. Goal may be complete.[/yellow]"})
            log.info("no_tasks_planned", cycle=self.cycles)
            return

        # Add new tasks to our tracking list
        for task in plan.tasks:
            # Check if task already exists (by id)
            existing = next((t for t in self.all_tasks if t.id == task.id), None)
            if not existing:
                self.all_tasks.append(task)

        self._emit({"type": "log", "message": "[bold yellow]Executing tasks in parallel...[/bold yellow]"})
        log.info("executing_tasks", cycle=self.cycles, num_tasks=len(plan.tasks))
        for task in plan.tasks:
            self._emit({"type": "task_start", "task_id": task.id, "description": task.description})
        
        tasks = [
            WorkerAgent(task=task, client=self.client).run()
            for task in plan.tasks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit({"type": "task_done", "task_id": task.id, "description": task.description, "status": "failed"})
                self._emit({"type": "log", "message": f"[red]Task {task.id} failed: {result}[/red]"})
                log.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                self._emit({"type": "task_done", "task_id": task.id, "description": task.description, "status": "completed"})
                self._emit({"type": "log", "message": f"[green]Task {task.id} completed[/green]"})
                log.info("task_completed", task_id=task.id, result=result[:100] if result else "")

        self._emit({"type": "log", "message": "[bold yellow]Testing...[/bold yellow]"})
        log.info("testing", cycle=self.cycles)
        test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._emit({"type": "test", "passed": True, "summary": test_result.summary, "failures": []})
            self._emit({"type": "log", "message": f"[green]Tests passed: {test_result.summary}[/green]"})
            log.info("tests_passed", summary=test_result.summary)
        else:
            self._emit({"type": "test", "passed": False, "summary": test_result.summary, "failures": test_result.failures})
            self._emit({"type": "log", "message": f"[red]Tests failed: {test_result.summary}[/red]"})
            for failure in test_result.failures:
                self._emit({"type": "log", "message": f"  • {failure}"})
            self._emit({"type": "log", "message": "[yellow]Will attempt fix in next cycle.[/yellow]"})
            log.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            # Append failures to goal instead of replacing
            self.goal = f"{self.goal}\\n\\nFix failing tests:\\n" + "\\n".join(test_result.failures)

    def _build_context(self) -> str:
        parts = []
        if self.all_tasks:
            parts.append("Previous tasks:")
            for t in self.all_tasks:
                parts.append(f"  {t.id}: {t.description} [{t.status}]")
        files = self.client.list_files(settings.workspace)
        if files:
            parts.append(f"\\nProject files ({len(files)}):")
            for f in files[:50]:  # Limit to 50 files
                parts.append(f"  {f}")
            if len(files) > 50:
                parts.append(f"  ... and {len(files) - 50} more")
        return "\\n".join(parts) if parts else "No previous tasks or files."

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        if failed > 0:
            return False
        if pending > 0:
            return False
        return completed >= len(self.all_tasks)
