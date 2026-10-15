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
         """Run the furrow-planner subagent to create a plan."""
         prompt = f"""You are a Furrow planner. Break the goal into parallelizable tasks.

Goal: {self.goal}

Return JSON only (no markdown, no explanation) with this shape:
{{
  "tasks": [
    {{
      "id": "1",
      "description": "Implement user authentication with JWT",
      "files": ["src/auth.py", "tests/test_auth.py"],
      "dependencies": []
    }}
  ],
  "rationale": "Brief explanation of the plan"
}}

If the goal is too large for one cycle, say so in rationale and break it into the most critical first slice."""
         
         result = await self._invoke_subagent("furrow-planner", prompt)
         return result

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

         result = await self._invoke_subagent("furrow-worker", prompt)
         return result

     async def _run_tester(self, tasks: list[Any]) -> Any:
         """Run the furrow-tester subagent to test completed work."""
         from furrow.config import TestResult
         
         # Format tasks for the tester prompt
         tasks_summary = []
         for task in tasks:
             tasks_summary.append(f"- {task.id}: {task.description} (Status: {task.status})")
         tasks_text = "\n".join(tasks_summary) if tasks_summary else "No tasks"
         
         prompt = f"""You are a tester agent in an autonomous coding system called Furrow.

Goal: {self.goal}

Completed Tasks:
{tasks_text}

Run tests, linting, and type checks on the completed work. Then determine if tests passed or failed.

Return JSON only (no markdown) with this shape:
{{
  "passed": true,
  "summary": "All tests passed",
  "failures": []
}}

If tests failed, list each failure in the failures array."""

         result = await self._invoke_subagent("furrow-tester", prompt)
         try:
             data = json.loads(result)
             return TestResult(**data)
         except (json.JSONDecodeError, ValueError):
             # Fallback: try to determine pass/fail from text
             passed = "passed" in result.lower()
             return TestResult(
                 passed=passed,
                 summary=result,
                 failures=[] if passed else [result]
             )

     async def _invoke_subagent(self, agent_type: str, prompt: str) -> str:
         """Invoke a subagent of the given type with the prompt and return its result."""
         if task is None:
             raise RuntimeError("Task tool not available")
         try:
             task_result = await self._call_task_tool(agent_type, prompt)
             return task_result.task_result
         except Exception as e:
             console.print(f"[red]Error invoking {agent_type} subagent: {e}[/red]")
             raise

     async def _call_task_tool(self, agent_type: str, prompt: str) -> Any:
         """Call the task tool to spawn a subagent."""
         return await task(
             description=f"Furrow {agent_type}",
             prompt=prompt,
             subagent_type=agent_type
         )
