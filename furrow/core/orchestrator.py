from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
logger = structlog.get_logger(__name__)

# Context compaction settings
MAX_CONTEXT_CHARS = 50000
COMPACTION_SUMMARY_PROMPT = """Summarize the following development history concisely, preserving:
- Key decisions and rationale
- Completed tasks and their outcomes
- Current blockers or failures
- Next steps needed

History:
{history}

Return a concise summary (max 2000 chars)."""


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.cycle_tasks: list[TaskModel] = []
        self.context_history: list[str] = []

    async def run(self) -> None:
        logger.info("orchestrator_start", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                logger.info("orchestrator_complete", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        logger.info("cycle_start", cycle=self.cycles)
        # Check and compact context if needed
        await self._maybe_compact_context()
        
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("cycle_no_tasks", cycle=self.cycles)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        self.cycle_tasks = plan.tasks
        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as status:
            test_result = await self.tester.run(self.goal, self.all_tasks)

        # Record cycle summary for context
        cycle_summary = self._create_cycle_summary(plan, test_result)
        self.context_history.append(cycle_summary)

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            # Mark all completed tasks as verified
            for task in self.all_tasks:
                if task.status == "completed":
                    task.status = "verified"
        else:
            logger.warning("tests_failed", failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures)

    def _create_cycle_summary(self, plan: Plan, test_result: TestResult) -> str:
        """Create a summary of this cycle for context history."""
        task_summaries = []
        for task in plan.tasks:
            status = task.status
            desc = task.description[:100]
            task_summaries.append(f"  - {task.id}: [{status}] {desc}")
        
        test_status = "PASSED" if test_result.passed else "FAILED"
        test_summary = test_result.summary[:200]
        
        return f"Cycle {self.cycles}:\n" + "\n".join(task_summaries) + f"\n  Tests: {test_status} - {test_summary}"

    async def _maybe_compact_context(self) -> None:
        """Compact context history if it exceeds the maximum size."""
        total_chars = sum(len(h) for h in self.context_history)
        if total_chars > MAX_CONTEXT_CHARS:
            logger.info("compacting_context", current_chars=total_chars, max_chars=MAX_CONTEXT_CHARS)
            history_text = "\n\n".join(self.context_history)
            prompt = COMPACTION_SUMMARY_PROMPT.format(history=history_text)
            try:
                summary = await self.client.complete(prompt, model=self.client.settings.model)
                self.context_history = [f"[COMPACTED HISTORY]\n{summary}"]
                logger.info("context_compacted", new_chars=len(summary))
            except Exception as e:
                logger.warning("context_compaction_failed", error=str(e))
                # Fallback: keep only recent cycles
                self.context_history = self.context_history[-5:]

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        
        completed = sum(1 for t in self.all_tasks if t.status in ("completed", "verified"))
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        pending = sum(1 for t in self.all_tasks if t.status == "pending")
        
        if failed > 0 and pending == 0:
            # All tasks that could be attempted have been attempted but some failed
            logger.error("orchestrator_failed", failed_count=failed)
            console.print("[bold red]Some tasks failed permanently. Halting.[/bold red]")
            return True
            
        if completed == len(self.all_tasks) and pending == 0:
            return True
            
        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
