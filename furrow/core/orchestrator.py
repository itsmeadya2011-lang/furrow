from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

# Import task tool for spawning subagents
try:
    from kilo import task
except ImportError:
    # Fallback for when kilo is not available (e.g., during testing)
    task = None

console = Console()


class Orchestrator:
     def __init__(self, goal: str, client: LLMClient | None = None) -> None:
         self.goal = goal
         self.client = client or LLMClient()
         self.settings = self.client.settings
         self.planner = PlannerAgent(client=self.client)
         self.cycles = 0
         self._tasks: list[Any] = []

    async def run(self) -> None:
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            
            # Check cycle limit
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(f"[yellow]Reached maximum cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                break
                
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

     async def _cycle(self) -> None:
         # Plan phase: use furrow-planner subagent
         with Status("[bold yellow]Planning...", console=console) as status:
             plan_result = await self._run_planner()
             plan = self._parse_plan(plan_result)
         console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

         if not plan.tasks:
             console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
             return

         # Execute phase: spawn furrow-worker subagents in parallel
         with Status("[bold yellow]Executing tasks in parallel...", console=console):
             tasks = [
                 self._run_worker(task)
                 for task in plan.tasks
             ]
             results = await asyncio.gather(*tasks, return_exceptions=True)

         # Update task statuses and store tasks
         self._tasks = []
         for task, result in zip(plan.tasks, results):
             if isinstance(result, Exception):
                 task.status = "failed"
                 task.result = str(result)
                 console.print(f"[red]Task {task.id} failed: {result}[/red]")
             else:
                 task.status = "completed"
                 task.result = result
                 console.print(f"[green]Task {task.id} completed[/green]")
             self._tasks.append(task)

         # Test phase: use furrow-tester subagent
         with Status("[bold yellow]Testing...", console=console) as status:
             test_result = await self._run_tester(plan.tasks)

         if test_result.passed:
             console.print(f"[green]Tests passed: {test_result.summary}[/green]")
         else:
             console.print(f"[red]Tests failed: {test_result.summary}[/red]")
             for failure in test_result.failures:
                 console.print(f"  • {failure}")
             console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
             # Prepend test failures to goal but preserve original goal
             if test_result.failures:
                 self.goal = f"Fix failing tests:\n" + "\n".join(test_result.failures) + f"\n\nOriginal goal:\n{self.goal}"

     def _is_done(self) -> bool:
         if not self._tasks:
             return False
         completed = sum(1 for t in self._tasks if t.status == "completed")
         failed = sum(1 for t in self._tasks if t.status == "failed")
         if failed > 0:
             return False
         if completed >= len(self._tasks):
             return True
         return False

     def _get_tasks(self) -> list[Any]:
         return self._tasks

     async def _run_planner(self) -> str:
         """Run the planner to create a plan."""
         plan = await self.planner.plan(self.goal)
         return plan.model_dump_json()

     def _parse_plan(self, plan_result: str) -> Plan:
         """Parse the planner result into a Plan object."""
         import json
         from furrow.config import Plan
         
         try:
             data = json.loads(plan_result)
             return Plan(**data)
         except (json.JSONDecodeError, ValueError) as e:
             console.print(f"[red]Failed to parse plan: {e}[/red]")
             console.print(f"[red]Plan result: {plan_result}[/red]")
             # Return an empty plan
             return Plan(tasks=[], rationale="Failed to parse plan")

     async def _run_worker(self, task: Any) -> str:
         """Run a furrow-worker subagent to execute a task."""
         prompt = f"""You are a Furrow worker. Execute the assigned task completely and return results.

Task: {task.description}
Files to touch: {', '.join(task.files) if task.files else 'any'}

## Behavior
- Work only on the task assigned to you. Do not scope creep.
- Make minimal, targeted changes. Do not refactor unrelated code.
- Run tests/lint if your changes affect existing code.
- Return a concise result: what you changed, any issues, and whether the task is complete.
- Do not spawn further subagents."""

         if task is None:
             raise RuntimeError("Task tool not available")
         try:
             task_result = await self._call_task_tool("furrow-worker", prompt)
             return task_result.task_result
         except Exception as e:
             console.print(f"[red]Error invoking furrow-worker subagent: {e}[/red]")
             raise

     async def _run_tester(self, tasks: list[Any]) -> TestResult:
         """Run the tester to test completed work."""
         return await TesterAgent(client=self.client).run(self.goal, tasks)
