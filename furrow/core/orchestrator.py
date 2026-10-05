from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from rich.console import Console
from rich.errors import MarkupError
from rich.markup import escape, render
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient

console = Console()
logger = logging.getLogger(__name__)


def _safe_console_print(message: str) -> None:
    """Print ``message`` to the console, never letting a rendering error kill the run.

    The console is the only channel that can raise ``MarkupError`` (the plain-text path
    below has its own fallback). If something still slips through -- a residual unescaped
    token, a rich quirk on a weird string -- catch it, fall back to the raw text with markup
    disabled, and keep going. A long-running token-burning loop must not die on bad data
    from an external source.
    """
    try:
        console.print(message)
    except MarkupError:
        console.print(message, markup=False)


def _safe(value: object) -> str:
    """Render ``value`` as markup-safe text, whatever it is.

    ``rich.markup.escape`` requires a ``str`` and raises ``TypeError: expected string or
    bytes-like object`` for anything else -- including the ``Exception`` objects the
    failure branches below hold. Because that TypeError is raised while the f-string is
    being built, it happens in the *caller*, before ``_safe_console_print`` is ever
    reached: the run dies with a nonsense traceback that discards the real error. Worse,
    for a missing ``ANTHROPIC_API_KEY`` (surfaced by the planner as a ``ValueError``) that
    hides the single most common first-run failure behind an unrelated-looking crash.

    Converting to ``str`` here, once, means a future call site cannot forget it.
    """
    return escape(str(value))


class Orchestrator:
    MAX_STALLED_CYCLES = 3

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.original_goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.on_event = on_event
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self.last_test_passed = False
        self.test_failures: list[str] = []
        self._stalled_cycles = 0

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {escape(self.original_goal)}",
                title="Furrow",
            )
        )
        self._emit(f"Goal: {escape(self.original_goal)}")
        while True:
            max_cycles = getattr(self.client.settings, "max_cycles", 0)
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._emit(
                    f"[bold yellow]Cycle limit reached (max_cycles={max_cycles}). "
                    "Halting.[/bold yellow]"
                )
                break
            self.cycles += 1
            self._emit(f"[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            before = self._state_signature()
            if not await self._cycle():
                self._emit("[bold red]Halting: planning failed, cannot continue.[/bold red]")
                break
            if self._is_done():
                self._emit(
                    "[bold green]Goal complete: every task is done and tests pass. "
                    "Halting.[/bold green]"
                )
                break
            if self._state_signature() != before:
                self._stalled_cycles = 0
            else:
                self._stalled_cycles += 1
                if self._stalled_cycles >= self.MAX_STALLED_CYCLES:
                    self._emit(f"[yellow]{self._stall_report()}[/yellow]")
                    self._emit(
                        f"[bold yellow]Halting: no progress in {self._stalled_cycles} "
                        "consecutive cycles.[/bold yellow]"
                    )
                    break

    async def _cycle(self) -> bool:
        with Status("[bold yellow]Planning...", console=console):
            try:
                plan = await self.planner.plan(self._planning_goal())
            except Exception as exc:
                self._emit(f"[red]Planner failed: {_safe(exc)}[/red]")
                return False
        # Pretty() renders LLM-derived data, which is untrusted: wrap it in the resilient
        # printer so a stray markup-like token can never kill the cycle.
        _safe_console_print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        fresh = self._absorb_plan(plan)
        completed = sum(1 for t in self.tasks if t.status == "completed")
        self._emit(
            f"Plan: {len(fresh)} new task(s); {len(self.tasks)} total, {completed} completed."
        )
        if not fresh:
            self._emit("[yellow]No new tasks planned this cycle.[/yellow]")

        await self._execute()
        await self._run_tests()
        return True

    def _absorb_plan(self, plan: Plan) -> list[TaskModel]:
        taken = {t.id for t in self.tasks}
        fresh: list[TaskModel] = []
        for task in plan.tasks:
            if task.id in taken:
                new_id = self._unique_id(task.id, taken)
                self._emit(
                    f"[yellow]Plan reused id '{escape(task.id)}'; new task registered as "
                    f"'{escape(new_id)}'.[/yellow]"
                )
                task.id = new_id
            task.status = "pending"
            task.result = None
            taken.add(task.id)
            self.tasks.append(task)
            fresh.append(task)
        return fresh

    def _unique_id(self, base: str, taken: set[str]) -> str:
        index = 2
        while f"{base}-{index}" in taken:
            index += 1
        return f"{base}-{index}"

    async def _execute(self) -> None:
        limit = getattr(self.client.settings, "max_parallel_tasks", 5)
        semaphore = asyncio.Semaphore(limit if limit > 0 else 1)
        while True:
            batch = self._ready_batch()
            if not batch:
                break
            with Status(f"[bold yellow]Executing {len(batch)} task(s)...", console=console):
                results = await asyncio.gather(
                    *(self._run_task(task, semaphore) for task in batch),
                    return_exceptions=True,
                )
            for task, result in zip(batch, results, strict=True):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                self._record(task, result)
        self._report_blocked()

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> str:
        worker = WorkerAgent(task=task, client=self.client)
        async with semaphore:
            return await worker.run()

    def _ready_batch(self) -> list[TaskModel]:
        by_id = {t.id: t for t in self.tasks}
        return [
            task
            for task in self.tasks
            if task.status == "pending"
            and all(
                dep not in by_id or by_id[dep].status == "completed" for dep in task.dependencies
            )
        ]

    def _record(self, task: TaskModel, result: object) -> None:
        if isinstance(result, BaseException):
            task.status = "failed"
            task.result = f"{type(result).__name__}: {result}"
            self._emit(f"[red]Task {_safe(task.id)} failed: {_safe(result)}[/red]")
            return
        task.status = "completed"
        task.result = result if isinstance(result, str) else str(result)
        self._emit(f"[green]Task {escape(task.id)} completed[/green]")

    def _report_blocked(self) -> None:
        by_id = {t.id: t for t in self.tasks}
        for task in self.tasks:
            if task.status != "pending":
                continue
            failed = [d for d in task.dependencies if d in by_id and by_id[d].status == "failed"]
            if failed:
                names = ", ".join(failed)
                self._emit(
                    f"[yellow]Task {escape(task.id)} skipped: dependency {escape(names)} did not "
                    "complete.[/yellow]"
                )

    async def _run_tests(self) -> None:
        try:
            with Status("[bold yellow]Testing...", console=console):
                result = await TesterAgent(client=self.client).run(self.original_goal, self.tasks)
        except Exception as exc:
            self.last_test_passed = False
            self.test_failures = [f"Test run raised {type(exc).__name__}: {exc}"]
            self._emit(f"[red]Test run raised an error: {_safe(exc)}[/red]")
            return
        self.last_test_passed = result.passed
        self.test_failures = list(result.failures)
        if result.passed:
            self._emit(f"[green]Tests passed: {escape(result.summary)}[/green]")
            return
        self._emit(f"[red]Tests failed: {escape(result.summary)}[/red]")
        for failure in result.failures:
            # Test output is untrusted (it comes from the test runner / an LLM verdict), so
            # escape it before the rich renderer sees it; _safe_console_print adds a second
            # net in case anything else in the line is mis-parsed.
            _safe_console_print(f"  • {escape(failure)}")
        self._emit("[yellow]Will attempt a fix in the next cycle.[/yellow]")

    def _planning_goal(self) -> str:
        failed_tasks = [t.id for t in self.tasks if t.status == "failed"]
        if not failed_tasks and not self.test_failures:
            return self.original_goal
        parts = [
            self.original_goal,
            "",
            (
                "The previous cycle made no clean finish. Continue towards the goal above "
                "while addressing the following, without dropping the goal itself:"
            ),
        ]
        if failed_tasks:
            parts.append(f"- Tasks that failed: {', '.join(failed_tasks)}")
        if self.test_failures:
            parts.append(f"- Test summary: {self.test_failures[0]}")
            parts.extend(f"- Test failure: {f}" for f in self.test_failures[1:])
        return "\n".join(parts)

    def _is_done(self) -> bool:
        if not self.tasks:
            return False
        if not all(t.status == "completed" for t in self.tasks):
            return False
        return self.last_test_passed

    def _state_signature(self) -> tuple[frozenset[str], frozenset[str], tuple[str, ...]]:
        """Fingerprint of what the run has *achieved*, for the stall guard.

        It deliberately describes the outcome -- which work is finished, which is broken,
        and what the suite currently reports -- instead of the task list. ``_absorb_plan``
        registers a fresh id for every re-planned task, so the list grows by at least one
        entry per cycle and a signature keyed on it differs on *every* cycle. That made
        this guard structurally unreachable, so ``max_cycles=0`` (unlimited, the default)
        kept re-planning the same doomed work forever: 78 cycles and 233 LLM calls in 8
        seconds, task ids ``1``, ``1-2``, ``1-3``, ... with ``_stalled_cycles`` stuck at 0.

        Work is therefore keyed on the task *description* rather than its id: the planner
        re-issues identical work under a new id each cycle, and doing the same work twice
        is not progress. ``PlannerAgent`` always gives a task a non-empty description
        (it synthesises one when the model sends none), so this key is never blank.
        Genuine progress -- new work completed, a different failure signature -- still
        moves the fingerprint, so the guard stays quiet on a run that is legitimately
        improving, however slowly.
        """
        completed = frozenset(t.description for t in self.tasks if t.status == "completed")
        failed = frozenset(t.description for t in self.tasks if t.status == "failed")
        return (completed, failed, tuple(self.test_failures))

    def _stall_report(self) -> str:
        """One line naming the state the run got stuck in, so the halt is actionable."""
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        last = self.test_failures[0] if self.test_failures else "no test failures reported"
        return (
            f"Stuck after {completed} completed and {failed} failed task(s); "
            f"last test result: {_safe(last)}. The last "
            f"{self._stalled_cycles} cycle(s) re-ran work already attempted without moving "
            "the outcome. Narrow the goal, or raise Orchestrator.MAX_STALLED_CYCLES to keep "
            "trying."
        )

    def _emit(self, message: str) -> None:
        # ``message`` mixes two very different kinds of content: furrow's own intentional
        # rich styling (the [red]/[/bold] tags we author here) and interpolated user/task
        # data (the goal, task ids, test failures) that can arrive from an untrusted remote
        # input -- a web form, a CLI argument, or an LLM response. Rich parses the whole
        # string as markup, so a literal "[/]" in the goal raises MarkupError inside
        # console.print() BEFORE the plain-text fallback below is ever reached.
        #
        # The fix is to escape *only* the interpolated data before it is spliced in (callers
        # wrap every user/LLM-derived token in escape()), which keeps furrow's own tags live
        # on the console while guaranteeing every user character round-trips intact:
        # escape("[/]") -> "\[/]", which rich renders back to "[/]" (verified for brackets,
        # backslashes and newlines). See run()/_cycle()/_absorb_plan()/_record()/
        # _report_blocked()/_run_tests().
        _safe_console_print(message)
        if self.on_event is None:
            return
        # Plain-text channel: strip the styling so the UI gets readable text. This fallback
        # is now genuinely reachable because console printing no longer raises on bad markup
        # (the interpolated data is escaped up front, and _safe_console_print swallows any
        # residual error). When it does trigger, fall back to the raw message verbatim so the
        # user's characters are never lost to the on_event consumers (e.g. the web UI).
        try:
            plain = render(message).plain.strip()
        except MarkupError:
            plain = message.strip()
        try:
            self.on_event(plain)
        except Exception as exc:
            logger.warning("on_event callback raised; event dropped: %s", exc)
