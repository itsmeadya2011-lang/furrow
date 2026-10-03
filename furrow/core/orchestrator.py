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
from furrow.config import Plan, TaskModel, TaskStatus, TestResult
from furrow.llm import LLMClient

console = Console()

# Bound on remembered task summaries so the planner prompt cannot grow without
# limit during a long-running loop.
MAX_REMEMBERED_SUMMARIES = 20
# Per-summary character budget inside the planner prompt.
MAX_SUMMARY_CHARS = 300
# Markers used to recognise a tester run that never actually ran anything.
_NO_RUNNER_MARKERS = ("no test runner", "no tests found", "test runner not found")


def _render_text(renderable: Panel | Pretty) -> str:
    """Render a rich renderable to plain text without echoing it to the console.

    ``Console.render_str`` only accepts markup strings, so a capture is used
    instead.
    """
    with console.capture() as capture:
        console.print(renderable)
    return capture.get()


class Orchestrator:
    """Drives the ``plan -> execute -> test`` loop until there is a reason to stop.

    The original ``goal`` is immutable for the whole run. What changes each cycle is
    :attr:`current_focus`: the goal, what has already been accomplished, and the
    latest test failures, so the planner keeps working the same objective instead of
    drifting onto whatever failed last.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal  # never reassigned: the objective must not drift
        self.current_focus = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.on_event = on_event
        self.cycles = 0
        self.plan: Plan | None = None
        self.tasks: list[TaskModel] = []
        self.completed_summaries: list[str] = []
        self._last_test_passed: bool | None = None
        self._last_test_result: TestResult | None = None
        self._idle_cycles = 0
        self._last_plan_had_no_tasks = False

    # ------------------------------------------------------------------ output

    def _emit(self, message: str | Panel | Pretty) -> None:
        """Print ``message`` and forward it to the ``on_event`` hook.

        ``message`` may be a rich renderable (``Panel``/``Pretty``) or markup text.
        The hook is typed ``Callable[[str], None]``, so a renderable is rendered to
        plain text before it leaves the loop; a web client should strip any markup.
        A broken consumer must never take the loop down with it.
        """
        console.print(message)
        if self.on_event is None:
            return
        try:
            text = message if isinstance(message, str) else _render_text(message)
            self.on_event(text)
        except Exception as exc:  # a bad consumer is not fatal
            console.print(f"[red]Event callback failed: {exc}[/red]")

    # --------------------------------------------------------------------- run

    async def run(self) -> None:
        header = Panel.fit(
            f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
            title="Furrow",
        )
        self._emit(header)
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            try:
                await self._cycle()
            except Exception as exc:  # one bad cycle is not fatal
                self._idle_cycles += 1
                self._emit(f"[red]Cycle {self.cycles} aborted: {exc}[/red]")
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break
            # Belt and braces: honour the cap even if _is_done misjudged.
            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._emit(
                    f"[bold yellow]Reached max_cycles ({max_cycles}). Halting.[/bold yellow]"
                )
                break
        self._emit(self._final_report())

    def _final_report(self) -> Panel:
        counts = {status: 0 for status in TaskStatus}
        for task in self._get_tasks():
            counts[task.status] += 1
        breakdown = ", ".join(f"{count} {status.value}" for status, count in counts.items())
        result = self._last_test_result
        if result is None:
            tests = "not run"
        elif self._tests_were_skipped(result):
            # Be honest: nothing ran, so this is neither a pass nor a failure.
            tests = "skipped (no test runner)"
        elif result.passed:
            tests = "passed"
        else:
            tests = f"failed: {result.summary}"
        return Panel(
            f"Cycles: {self.cycles}\n"
            f"Tasks: {breakdown or 'none'}\n"
            f"Tests: {tests}\n"
            f"Idle cycles: {self._idle_cycles}",
            title="Summary",
            border_style="cyan",
        )

    # ------------------------------------------------------------------- cycle

    async def _cycle(self) -> None:
        self.current_focus = self._build_focus(self._last_test_result)

        with Status("[bold yellow]Planning...", console=console):
            try:
                plan = await self.planner.plan(self.current_focus)
            except Exception as exc:  # unparseable plan is not fatal
                self._emit(f"[red]Planning failed: {exc}[/red]")
                self._record_progress(idle=True)
                return

        self.plan = plan
        self.tasks = plan.tasks
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._last_plan_had_no_tasks = not plan.tasks
        if self._last_plan_had_no_tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._record_progress(idle=True)
            return

        executed = await self._execute_tasks(plan.tasks)
        self._record_summaries(executed)

        test_result = await self._run_tests(plan)
        self._last_test_result = test_result
        if test_result is None:
            self._last_test_passed = None
        elif self._tests_were_skipped(test_result):
            # No runner available is neither a failure nor a pass: do not chase a
            # phantom red build, and never let a skip count as a green test gate.
            self._last_test_passed = None
            self._emit("[yellow]No test runner available; skipping test gate.[/yellow]")
        else:
            self._last_test_passed = test_result.passed
            if test_result.passed:
                self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
            else:
                self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
                for failure in test_result.failures:
                    self._emit(f"  • {failure}")
                self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")

        # No task ran (nothing to do, or everything was already done), or the
        # planner declared the goal achieved: that is an idle cycle.
        self._record_progress(idle=not executed or plan.complete is True)

    # --------------------------------------------------------------- execution

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[TaskModel]:
        """Run ``tasks`` in dependency-ordered, concurrency-limited waves.

        Returns the tasks that were actually executed successfully this cycle.
        """
        known_ids = {task.id for task in tasks}
        completed_ids: set[str] = set()
        broken_ids: set[str] = set()
        executed: list[TaskModel] = []

        for task in tasks:
            if task.status == TaskStatus.COMPLETED:
                completed_ids.add(task.id)
                self._emit(f"[dim]Task {task.id} already completed; skipping.[/dim]")
            else:
                task.status = TaskStatus.PENDING

        remaining = [task for task in tasks if task.id not in completed_ids]
        while remaining:
            doomed = [t for t in remaining if any(d in broken_ids for d in t.dependencies)]
            if doomed:
                for task in doomed:
                    self._block(task, "A dependency failed.")
                    remaining.remove(task)
                    broken_ids.add(task.id)
                continue

            wave = [
                task
                for task in remaining
                if all(dep in completed_ids or dep not in known_ids for dep in task.dependencies)
            ]
            if not wave:
                # Nothing left can ever become runnable: a dependency cycle.
                self._emit("[red]Dependency cycle detected in plan.[/red]")
                for task in remaining:
                    self._block(task, "Unresolvable dependencies (cycle).")
                remaining = []
                break

            for task in wave:
                remaining.remove(task)
            await self._run_wave(wave, completed_ids, broken_ids, executed)

        return executed

    async def _run_wave(
        self,
        wave: list[TaskModel],
        completed_ids: set[str],
        broken_ids: set[str],
        executed: list[TaskModel],
    ) -> None:
        limit = self.client.settings.max_parallel_tasks
        semaphore = asyncio.Semaphore(limit if limit > 0 else 1)

        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                task.status = TaskStatus.RUNNING
                self._emit(f"[yellow]▶ task {task.id}: {task.description}[/yellow]")
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks...", console=console):
            results = await asyncio.gather(
                *(run_task(task) for task in wave),
                return_exceptions=True,
            )

        for task, result in zip(wave, results, strict=True):
            # gather(return_exceptions=True) yields `str | BaseException`. Testing
            # against BaseException rather than Exception keeps asyncio.CancelledError
            # (a BaseException on 3.10+) a failure and narrows the success branch to str.
            if isinstance(result, BaseException):
                task.status = TaskStatus.FAILED
                task.result = f"{type(result).__name__}: {result}"
                broken_ids.add(task.id)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                completed_ids.add(task.id)
                executed.append(task)
                self._emit(f"[green]Task {task.id} completed[/green]")

    def _block(self, task: TaskModel, reason: str) -> None:
        task.status = TaskStatus.BLOCKED
        task.result = reason
        self._emit(f"[red]Task {task.id} blocked: {reason}[/red]")

    # ----------------------------------------------------------------- testing

    async def _run_tests(self, plan: Plan) -> TestResult | None:
        with Status("[bold yellow]Testing...", console=console):
            try:
                return await TesterAgent(client=self.client).run(self.goal, plan.tasks)
            except Exception as exc:  # a broken tester is not fatal
                self._emit(f"[red]Testing failed: {exc}[/red]")
                return None

    @staticmethod
    def _tests_were_skipped(test_result: TestResult) -> bool:
        # The explicit signal wins: a result flagged ``skipped`` never ran.
        if test_result.skipped:
            return True
        if test_result.passed:
            return False
        # Legacy text fallback: an older/no-signal result still gets recognised.
        haystack = " ".join([test_result.summary, *test_result.failures]).lower()
        return any(marker in haystack for marker in _NO_RUNNER_MARKERS)

    # -------------------------------------------------------------------- focus

    def _build_focus(self, test_result: TestResult | None) -> str:
        sections = [f"ORIGINAL GOAL (this never changes):\n{self.goal}"]

        if self.completed_summaries:
            done = "\n".join(f"- {item}" for item in self._summaries_for_prompt())
            sections.append(f"ALREADY COMPLETED - do not redo this work:\n{done}")

        if test_result is not None and not test_result.passed:
            failures = test_result.failures or [test_result.summary]
            body = "\n".join(f"- {failure}" for failure in failures)
            sections.append(f"Fix these failing tests:\n{body}")

        return "\n\n".join(sections)

    def _record_summaries(self, executed: list[TaskModel]) -> None:
        for task in executed:
            summary = (task.result or task.description).strip()
            entry = f"[{task.id}] {summary}"
            if entry not in self.completed_summaries:
                self.completed_summaries.append(entry)
        if len(self.completed_summaries) > MAX_REMEMBERED_SUMMARIES:
            self.completed_summaries = self.completed_summaries[-MAX_REMEMBERED_SUMMARIES:]

    def _summaries_for_prompt(self) -> list[str]:
        return [item[:MAX_SUMMARY_CHARS] for item in self.completed_summaries]

    # ------------------------------------------------------------- termination

    def _record_progress(self, *, idle: bool) -> None:
        if idle:
            self._idle_cycles += 1
        else:
            self._idle_cycles = 0

    def _is_done(self) -> bool:
        max_cycles = self.client.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            return True
        if self._last_plan_had_no_tasks:
            return True
        if self.plan is not None and self.plan.complete and self._last_test_passed is not False:
            # The planner says the goal is achieved. Never honour that while the
            # test suite is known to be red; a missing test signal is tolerated so
            # a project without a runner can still finish.
            return True
        max_idle = self.client.settings.max_idle_cycles
        return max_idle > 0 and self._idle_cycles >= max_idle

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks
