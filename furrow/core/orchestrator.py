"""The development loop: plan -> execute -> test -> repeat until the goal is met.

The loop is unbounded by design (``max_cycles == 0``), so every exit is explicit:
the goal is proved by a green tester plus a clean task board, or one of the
guards below trips and ``stop_reason`` explains why.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import re
from collections.abc import Callable, Iterator
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.pretty import Pretty

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.config import settings as default_settings
from furrow.llm import LLMClient

try:  # structlog is a declared dependency; the fallback only guards odd installs.
    import structlog
except ModuleNotFoundError:  # pragma: no cover - logging must never break the loop
    structlog = None  # type: ignore[assignment]

console = Console()

# A runaway LLM or a genuinely broken repo must not spin forever: these budgets
# are the difference between "keeps working" and "hammers the API all night".
MAX_CONSECUTIVE_PLANNER_FAILURES = 3
MAX_IDENTICAL_FAILURES = 3

# A dependency in one of these states can never be satisfied, so dependents are skipped.
BLOCKED_STATUSES = frozenset({"failed", "skipped"})

LOGGER_NAME = "furrow.orchestrator"
_logging_configured = False


def one_line(value: object, limit: int = 200) -> str:
    """Collapse arbitrary text (multi-line LLM output included) into one clean event line.

    ``on_event`` consumers (web UI, log scrapers) assume a single line with no
    markup and no surrounding whitespace, so every emitted string goes through here.
    """
    text = " ".join(str(value).split())
    if len(text) > limit:
        text = text[: max(limit - 3, 1)] + "..."
    return text


def _configure_logging(level: str) -> None:
    """Configure structlog once, defensively, at the orchestrator's log level."""
    global _logging_configured
    if _logging_configured or structlog is None:
        return
    _logging_configured = True
    # Logging is best effort: a structlog build that rejects this configuration must
    # cost us log lines, never the loop itself.
    with contextlib.suppress(Exception):
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(colors=False),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, str(level).upper(), logging.INFO)
            ),
            cache_logger_on_first_use=True,
        )


_DURATION_RE = re.compile(r'\b(?:in\s+)?\d+\.\d+s\b|\b\d+s\b')
_PATH_RE = re.compile(r'(?:/[\w.-]+){2,}|[A-Za-z]:[/\\][\w.\\-]+')
_HEX_RE = re.compile(r'0x[0-9a-fA-F]+')
_PID_RE = re.compile(r'\[(?:\d{3,6})\]|\(pid\s+\d+\)|\bpid=\d+\b')
_UUID_RE = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
_QUOTED_TIMING_RE = re.compile(r'"\d+\.\d+s?"')
_WS_RE = re.compile(r'\s+')


def _normalize_failure(text: str) -> str:
    """Strip cosmetic churn from one failure line so the assertion identity remains."""
    text = _DURATION_RE.sub('', text)
    text = _PATH_RE.sub('', text)
    text = _HEX_RE.sub('', text)
    text = _PID_RE.sub('', text)
    text = _UUID_RE.sub('', text)
    text = _QUOTED_TIMING_RE.sub('', text)
    text = _WS_RE.sub(' ', text).strip()
    return text


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


class Orchestrator:
    """Drives planner/worker/tester agents in cycles until the goal is complete.

    ``goal`` is immutable: the original objective lives in :attr:`goal` forever, while
    :attr:`focus` carries the current refinement (e.g. the latest test failures) that
    the planner is steered with. Keeping them apart is what stops the loop from
    forgetting what it was asked to do.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: Callable[[str], None] | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.focus = goal
        self.settings = settings if settings is not None else default_settings
        # A caller-supplied client owns its own settings (models, keys); we only
        # borrow its transport, so the orchestrator settings drive loop behaviour.
        self.client = client or LLMClient()
        self.on_event = on_event
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.stop_reason = ""
        self.last_test_result: TestResult | None = None
        self._tasks: list[TaskModel] = []
        self._planner_failures = 0
        self._stagnant_cycles = 0
        self._last_failure_set: set[str] | None = None
        _configure_logging(self.settings.log_level)

    # ------------------------------------------------------------------ public

    async def run(self) -> None:
        """Run cycles until the goal is proved or a guard trips. Never raises for LLM errors."""
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {escape(one_line(self.goal))}",
                title="Furrow",
            )
        )
        self._log("info", "orchestrator.start", goal=self.goal, max_cycles=self.settings.max_cycles)

        while True:
            if self._cycle_budget_exhausted():
                self._stop(f"max cycles reached ({self.settings.max_cycles})")
                break
            self.cycles += 1
            self._log("info", "cycle.start", cycle=self.cycles, focus=self.focus)
            if not await self._cycle():
                # _cycle() already recorded the stop reason (planner/tester guards).
                break
            if self._is_done():
                self._stop("goal complete", success=True)
                break

        self._log(
            "info",
            "orchestrator.stop",
            cycles=self.cycles,
            stop_reason=self.stop_reason,
            done=self.is_done,
        )
        if self.is_done:
            console.print(f"[bold green]Goal complete after {self.cycles} cycle(s).[/bold green]")
        else:
            console.print(
                f"[bold yellow]Stopped after {self.cycles} cycle(s): "
                f"{escape(self.stop_reason)}[/bold yellow]"
            )

    @property
    def is_done(self) -> bool:
        """True only when a real tester run passed and the board has no failed work left."""
        return self._is_done()

    @property
    def tasks(self) -> list[TaskModel]:
        """The tasks of the most recent plan with their recorded statuses and results."""
        return self._get_tasks()

    # ------------------------------------------------------------------- cycle

    async def _cycle(self) -> bool:
        """Run one plan/execute/test cycle. Returns False when the loop must stop."""
        self._emit(f"Cycle {self.cycles}: planning")
        try:
            with self._spinner("[bold yellow]Planning..."):
                plan = await self.planner.plan(
                    f"Original goal: {self.goal}\nCurrent focus: {self.focus}"
                )
        except Exception as exc:
            # A malformed planner reply is an expected, recoverable event: keep going
            # but give up after MAX_CONSECUTIVE_PLANNER_FAILURES so we do not spin.
            self._planner_failures += 1
            self._log(
                "error",
                "planner.failed",
                cycle=self.cycles,
                error=str(exc),
                consecutive_failures=self._planner_failures,
            )
            self._emit(f"Cycle {self.cycles}: planner failed ({one_line(exc)})", style="red")
            if self._planner_failures >= MAX_CONSECUTIVE_PLANNER_FAILURES:
                self._stop(f"planner failed {MAX_CONSECUTIVE_PLANNER_FAILURES} cycles in a row")
                return False
            return True
        self._planner_failures = 0

        if not plan.tasks:
            self._emit(f"Cycle {self.cycles}: planner returned no tasks", style="yellow")
            # The planner has nothing left to give, but that is not proof of success,
            # so we stop and say exactly that rather than claiming the goal is met.
            self._stop("planner returned no tasks")
            return False

        # A new plan means a new unit of work: statuses restart so dependency waves
        # are recomputed from scratch for this cycle.
        self._tasks = list(plan.tasks)
        for task in self._tasks:
            task.status = "pending"
            task.result = None
        self._emit(f"Cycle {self.cycles}: plan ready with {len(self._tasks)} task(s)")
        self._log("info", "plan.ready", cycle=self.cycles, tasks=len(self._tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        await self._execute(self._tasks)
        return await self._test(self._tasks)

    async def _execute(self, tasks: list[TaskModel]) -> None:
        """Run the plan in dependency waves: independent tasks go out in parallel."""
        index: dict[str, TaskModel] = {}
        for task in tasks:
            index.setdefault(task.id, task)

        for task in self._dependency_cycle_members(index):
            self._skip(task, "circular dependency")

        cap = self.settings.max_parallel_tasks
        # A cap of 0 or less means "no cap"; a Semaphore(0) would deadlock the wave.
        semaphore = asyncio.Semaphore(cap) if cap > 0 else None

        while True:
            pending = [task for task in tasks if task.status == "pending"]
            if not pending:
                return

            runnable: list[TaskModel] = []
            for task in pending:
                deps = [index[dep] for dep in task.dependencies if dep in index]
                if any(dep.status in BLOCKED_STATUSES for dep in deps):
                    self._skip(task, "dependency did not complete")
                    continue
                if all(dep.status == "completed" for dep in deps):
                    runnable.append(task)

            if not runnable:
                # Nothing left can be unlocked (unresolvable graph): skip the rest
                # instead of looping on a wave that can never start.
                for task in pending:
                    if task.status == "pending":
                        self._skip(task, "dependencies never completed")
                return

            for task in runnable:
                task.status = "in_progress"
            self._emit(f"Cycle {self.cycles}: running {len(runnable)} task(s)")
            results = await asyncio.gather(
                *(self._run_task(task, semaphore) for task in runnable),
                return_exceptions=True,  # one failing task must not abort its siblings
            )

            for task, result in zip(runnable, results, strict=True):
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = f"{type(result).__name__}: {result}"
                    self._log(
                        "error", "task.failed", cycle=self.cycles, task=task.id, error=str(result)
                    )
                    self._emit(f"Cycle {self.cycles}: task {task.id} failed", style="red")
                else:
                    task.status = "completed"
                    task.result = result
                    self._log("info", "task.completed", cycle=self.cycles, task=task.id)
                    self._emit(f"Cycle {self.cycles}: task {task.id} completed", style="green")

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore | None) -> str:
        kwargs = self._worker_kwargs(task)
        if semaphore is None:
            return await WorkerAgent(task=task, client=self.client, **kwargs).run()
        async with semaphore:
            return await WorkerAgent(task=task, client=self.client, **kwargs).run()

    def _worker_kwargs(self, task: TaskModel) -> dict[str, Any]:
        """Constructor kwargs for a worker, carrying what its dependencies produced.

        The mapping is assembled per task at run time: by then every dependency is
        finished (that is the wave invariant) and the values are the current results,
        not a snapshot taken before the wave ran. It is passed as the single
        ``dependency_results`` argument the worker declares - splatting it as
        ``**{dep_id: result}`` would raise ``TypeError`` on every dependent task.
        ``WorkerAgent`` owns its constructor, so the optional kwarg is probed with
        ``inspect.signature`` instead of being assumed: a worker refactor that drops
        it must not break the orchestrator.
        """
        try:
            parameters = inspect.signature(WorkerAgent.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic constructors
            return {}
        if "dependency_results" not in parameters:
            return {}
        by_id = {other.id: other for other in self._tasks}
        completed = {
            dep_id: by_id[dep_id].result
            for dep_id in task.dependencies
            if dep_id in by_id
            and by_id[dep_id].status == "completed"
            and by_id[dep_id].result is not None
        }
        return {"dependency_results": completed}

    async def _test(self, tasks: list[TaskModel]) -> bool:
        """Judge the cycle with a real tester run. Returns False when the loop must stop."""
        self._emit(f"Cycle {self.cycles}: testing")
        try:
            with self._spinner("[bold yellow]Testing..."):
                result = await TesterAgent(client=self.client).run(self.goal, tasks)
        except Exception as exc:
            # A tester blow-up is a failed cycle, not a dead session.
            message = f"tester error: {exc}"
            result = TestResult(passed=False, summary=message, failures=[message])
            self._log("error", "tester.failed", cycle=self.cycles, error=str(exc))

        self.last_test_result = result
        self._log(
            "info",
            "tests.finished",
            cycle=self.cycles,
            passed=result.passed,
            summary=result.summary,
        )

        if result.passed:
            self._last_failure_set = None
            self._stagnant_cycles = 0
            self._emit(
                f"Cycle {self.cycles}: tests passed ({one_line(result.summary)})", style="green"
            )
            return True

        detail = (
            f"{len(result.failures)} failure{'' if len(result.failures) == 1 else 's'}"
            if result.failures
            else one_line(result.summary)
        )
        self._emit(f"Cycle {self.cycles}: tests failed ({detail})", style="red")
        for failure in result.failures:
            console.print(f"  • {escape(one_line(failure))}", style="red", markup=False)
        # Steer the next cycle with the failures; self.goal stays the original objective.
        if result.failures:
            self.focus = "Fix failing tests:\n" + "\n".join(result.failures)
        else:
            self.focus = (
                f"Continue working towards: {self.goal}\nLatest test output: {result.summary}"
            )
        console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
        return not self._no_progress(result)

    # -------------------------------------------------------------- loop state

    def _is_done(self) -> bool:
        """Goal met == the freshest tester run passed AND no task is failed/skipped/pending.

        Checking only "all tasks completed" is what made the old loop exit after a
        single cycle, and checking only the tests ignores tasks that never ran.
        """
        if self.last_test_result is None or not self.last_test_result.passed:
            return False
        tasks = self._get_tasks()
        if not tasks:
            return False
        return all(task.status == "completed" for task in tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return list(self._tasks)

    def _cycle_budget_exhausted(self) -> bool:
        """``max_cycles == 0`` means unlimited, which is why zero is not a budget."""
        limit = self.settings.max_cycles
        return limit > 0 and self.cycles >= limit

    def _no_progress(self, result: TestResult) -> bool:
        """True when similar failures have repeated often enough to give up.

        Unlike the old exact-fingerprint check, normalization collapses cosmetic
        churn (timings, paths, PIDs, hex addresses) before comparing cycles, so
        a loop whose failure text varies around the same assertion does not spin
        forever. A cycle with fewer failures or a clearly different failure set
        resets the counter, because fixing is never a reason to stop.
        """
        current = self._normalized_failure_set(result)
        threshold = self.settings.stagnation_threshold
        max_stagnant = self.settings.max_stagnant_cycles

        if not 0 < threshold <= 1:
            self._log("warning", "stagnation.invalid_threshold", threshold=threshold)
            threshold = 0.8
        if max_stagnant < 1:
            self._log("warning", "stagnation.invalid_max", max_stagnant=max_stagnant)
            max_stagnant = 3

        if self._last_failure_set is None:
            self._last_failure_set = current
            self._stagnant_cycles = 1
        elif len(current) < len(self._last_failure_set):
            self._last_failure_set = current
            self._stagnant_cycles = 1
        else:
            similarity = _jaccard(current, self._last_failure_set)
            if similarity < threshold:
                self._last_failure_set = current
                self._stagnant_cycles = 1
            else:
                self._stagnant_cycles += 1

        if self._stagnant_cycles >= max_stagnant:
            self._stop(f"no progress after {max_stagnant} identical failures")
            return True
        return False

    @staticmethod
    def _normalized_failure_set(result: TestResult) -> set[str]:
        """Order-insensitive set of normalized failure fingerprints.

        Normalization collapses cosmetic churn so two cycles with the same assertion
        but different timings, paths, PIDs or addresses produce the same fingerprint.
        """
        if result.failures:
            return {_normalize_failure(f) for f in result.failures if _normalize_failure(f)}
        return {_normalize_failure(result.summary)}

    def _dependency_cycle_members(self, index: dict[str, TaskModel]) -> list[TaskModel]:
        """Return the tasks caught in a dependency cycle (they can never be unlocked)."""
        state: dict[str, int] = {}  # 0 = unvisited, 1 = on stack, 2 = resolved
        stack: list[str] = []
        cyclic: set[str] = set()

        def visit(task_id: str) -> None:
            state[task_id] = 1
            stack.append(task_id)
            for dep in index[task_id].dependencies:
                if dep not in index:
                    # Unknown ids (earlier cycle, or planner noise) are treated as
                    # satisfied: a stale id must not freeze the whole plan.
                    continue
                if state.get(dep, 0) == 1:
                    cyclic.update(stack[stack.index(dep) :])
                elif state.get(dep, 0) == 0:
                    visit(dep)
            stack.pop()
            state[task_id] = 2

        for task_id in index:
            if state.get(task_id, 0) == 0:
                visit(task_id)
        return [index[task_id] for task_id in cyclic]

    def _skip(self, task: TaskModel, reason: str) -> None:
        task.status = "skipped"
        task.result = reason
        self._log("info", "task.skipped", cycle=self.cycles, task=task.id, reason=reason)
        self._emit(f"Cycle {self.cycles}: task {task.id} skipped ({reason})", style="yellow")

    def _stop(self, reason: str, *, success: bool = False) -> None:
        """Freeze the stop reason and publish the single terminal event."""
        self.stop_reason = reason
        if success:
            self._emit(f"Goal complete: {one_line(self.goal, 120)}", style="bold green")
        else:
            self._emit(f"Stopping: {reason}", style="bold yellow")
        self._log(
            "info",
            "orchestrator.stopping",
            reason=reason,
            cycles=self.cycles,
            success=success,
        )

    # ----------------------------------------------------------------- output

    def _emit(self, message: str, *, style: str | None = None) -> None:
        """Publish a single-line, markup-free event to the callback, the log and the console."""
        line = one_line(message)
        self._log("info", "event", line=line)
        if self.on_event is not None:
            try:
                self.on_event(line)
            except Exception:
                # A broken UI consumer must never take the loop down with it.
                self._log("warning", "on_event.failed", line=line)
        if style:
            console.print(line, style=style, markup=False, highlight=False)

    def _log(self, level: str, event: str, **fields: Any) -> None:
        """Structured logging that can never raise, whatever the handler does."""
        try:
            if structlog is not None:
                getattr(structlog.get_logger(LOGGER_NAME), level)(event, **fields)
            else:  # pragma: no cover - only when structlog is unavailable
                logging.getLogger(LOGGER_NAME).log(
                    getattr(logging, level.upper(), logging.INFO), event, extra={"fields": fields}
                )
        except Exception:  # pragma: no cover - logging is best effort by design
            pass

    @contextlib.contextmanager
    def _spinner(self, message: str) -> Iterator[None]:
        """Live spinner for humans; skipped when stdout is not a terminal (CI, tests)."""
        if console.is_terminal:
            with console.status(message, spinner="dots"):
                yield
        else:
            yield
