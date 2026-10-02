from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient

console = Console()

_RESULT_CHARS = 200

# Status used for a task that never ran. Deliberately neither "completed" nor
# "failed": nothing was attempted, and the user's cycle-end counters must not
# pretend otherwise.
SKIPPED = "skipped"


def _orderable(dependencies: dict[int, list[int]]) -> set[int]:
    """Return the positions that Kahn's algorithm can order.

    Whatever is *missing* from the result is stuck in -- or downstream of -- a
    dependency cycle, and can therefore never become runnable. Detecting that up
    front is what keeps a malformed LLM plan from wedging the scheduler: a cycle
    costs a cycle, not an infinite loop.
    """
    indegree = {position: len(deps) for position, deps in dependencies.items()}
    dependents: dict[int, list[int]] = {position: [] for position in dependencies}
    for position, deps in dependencies.items():
        for dep in deps:
            dependents[dep].append(position)
    ready = [position for position, degree in indegree.items() if degree == 0]
    orderable: set[int] = set()
    while ready:
        position = ready.pop()
        orderable.add(position)
        for dependent in dependents[position]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
    return orderable


class Orchestrator:
    """Drives the plan -> execute -> test loop until it is told to stop.

    The loop only halts for one of three reasons:

    * the planner has nothing left to plan (``_is_done``),
    * tests have failed ``settings.max_consecutive_failures`` cycles in a row,
    * the ``settings.max_cycles`` budget is exhausted (``0`` means unlimited).

    It deliberately does *not* halt just because one cycle succeeded -- that is
    the whole point of Furrow.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.original_goal = goal
        self.current_goal = goal
        self.tasks: list[TaskModel] = []
        self.cycles = 0
        self.consecutive_failures = 0
        self.tasks_planned_last_cycle = 0
        self.on_event: Callable[[dict[str, Any]], None] | None = None
        # An explicit Settings wins over the global one, but the global is never
        # replaced by ``None``.
        self.settings: Settings = settings if settings is not None else global_settings
        # NOTE: never pass ``None`` in here. ``LLMClient.__init__`` only falls back
        # to the global settings for an *omitted* argument; an explicit ``None``
        # would set ``client.settings = None`` and break every ``.settings.*``
        # lookup further down.
        self.client = client or LLMClient(settings=self.settings)

    @property
    def goal(self) -> str:
        """The goal handed to the agents this cycle (kept for backwards compat)."""
        return self.current_goal

    @goal.setter
    def goal(self, value: str) -> None:
        self.current_goal = value

    async def run(self) -> None:
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.original_goal}",
                title="Furrow",
            )
        )
        sem = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            self._emit(
                {
                    "type": "cycle_start",
                    "goal": self.current_goal,
                    "focus": self._focus(),
                    "tasks_tracked": len(self.tasks),
                }
            )
            await self._cycle(sem)

            if self._is_done():
                self._halt("goal complete: the planner has no work left to plan", style="green")
                break

            max_failures = self.settings.max_consecutive_failures
            if max_failures > 0 and self.consecutive_failures >= max_failures:
                self._halt(
                    f"failure limit reached: {self.consecutive_failures} consecutive cycles "
                    f"with failing tests (max_consecutive_failures={max_failures})",
                    style="red",
                )
                break

            max_cycles = self.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._halt(f"cycle budget reached: {self.cycles}/{max_cycles} cycles")
                break

    async def _cycle(self, sem: asyncio.Semaphore | None = None) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            status.update("Planning...")
            plan: Plan = await self.planner().plan(self.current_goal)
            status.update(f"Planned {len(plan.tasks)} task(s)")
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._merge_tasks(plan.tasks)
        self.tasks_planned_last_cycle = len(plan.tasks)
        self._emit(
            {
                "type": "plan",
                "task_ids": [task.id for task in plan.tasks],
                "count": len(plan.tasks),
                "rationale": plan.rationale,
                "tasks_tracked": len(self.tasks),
            }
        )

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        await self._execute(plan.tasks, sem)

        with Status("[bold yellow]Testing...", console=console) as status:
            status.update("Running tests...")
            test_result: TestResult = await self.tester().run(self.original_goal, plan.tasks)
            status.update(test_result.summary)

        if test_result.passed:
            self.consecutive_failures = 0
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self.consecutive_failures += 1
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # The original goal is never overwritten, otherwise a few failing
            # cycles would leave the agents optimising the wrong target.
            self.current_goal = (
                f"Original goal: {self.original_goal}\n\nFix these failing tests:\n"
                + "\n".join(test_result.failures)
            )

        self._emit(
            {
                "type": "test_result",
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": list(test_result.failures),
                "consecutive_failures": self.consecutive_failures,
            }
        )
        self._emit(
            {
                "type": "cycle_end",
                "goal": self.current_goal,
                "tasks_completed": sum(1 for t in self.tasks if t.status == "completed"),
                "tasks_failed": sum(1 for t in self.tasks if t.status == "failed"),
            }
        )

    async def _execute(
        self, tasks: list[TaskModel], sem: asyncio.Semaphore | None = None
    ) -> None:
        """Run the plan, honouring task dependencies and the parallelism cap.

        Why the dependency graph is load-bearing rather than decoration:
        ``WorkerAgent`` writes *whole* files through ``client.write_file``, so two
        workers editing the same file at the same time perform an unsynchronised
        read-modify-write -- both read version N, each produces its own N+1, the
        second write clobbers the first and the loser's edit disappears with no
        error anywhere. The planner is explicitly asked for ``dependencies`` so
        that such tasks never overlap, and this method is where that promise is
        actually kept.

        The scheduler is a ready-queue loop: launch every task whose dependencies
        have all *completed*, up to ``max_parallel_tasks``, await the next
        completion, re-evaluate. A plan with no dependencies degenerates to
        "keep the cap full", i.e. exactly the previous behaviour -- ordering must
        not cost parallelism, that would throw away the whole point of the thing.
        """
        if not tasks:
            return
        if sem is None:
            sem = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        cap = max(1, self.settings.max_parallel_tasks)
        total = len(tasks)

        async def run_one(task: TaskModel) -> Any:
            # The semaphore stays the hard ceiling on real concurrency; ``cap``
            # below only decides how much work we *queue* up, so that dependent
            # tasks are not even created until their turn is legitimate.
            async with sem:
                return await self.worker(task).run()

        # --- resolve the graph against THIS cycle's plan only -----------------
        # A dependency naming a task that is not in this plan (a typo, or a
        # reference to a task from an earlier cycle) can never be satisfied by
        # anything we are willing to wait for, so waiting on it would wedge the
        # run forever. Record the dangling reference and carry on.
        index_of: dict[str, int] = {}
        for position, task in enumerate(tasks):
            index_of.setdefault(task.id, position)

        deps: dict[int, list[int]] = {}
        dangling: dict[str, list[str]] = {}
        for position, task in enumerate(tasks):
            resolved: list[int] = []
            missing: list[str] = []
            for raw in task.dependencies:
                dep_id = str(raw).strip()
                if not dep_id:
                    continue
                if dep_id in index_of:
                    resolved.append(index_of[dep_id])
                else:
                    missing.append(dep_id)
            deps[position] = resolved
            if missing:
                dangling[task.id] = missing
        if dangling:
            self._dependency_warning(
                "unknown_dependency",
                "ignoring dependencies that name no task in this plan",
                details={"refs": dangling},
            )

        # --- cycles: report them, never spin on them -------------------------
        orderable = _orderable(deps)
        stuck = [position for position in range(total) if position not in orderable]
        if stuck:
            stuck_ids = sorted({tasks[position].id for position in stuck})
            self._dependency_warning(
                "cycle",
                "cycle in the plan's dependency graph; these tasks cannot run",
                details={"tasks": stuck_ids},
            )
            reason = (
                "unmet dependency: cycle in the plan's dependency graph "
                f"(blocked: {', '.join(stuck_ids)})"
            )
            for position in stuck:
                self._skip_task(tasks[position], reason)

        pending = set(orderable)
        state: dict[int, str] = {}
        futures: dict[asyncio.Future[Any], int] = {}
        done = 0

        with Status(
            f"[bold yellow]Executing {total} task(s)...", console=console
        ) as status:
            while True:
                # Launch everything eligible right now, up to the cap. Iterating
                # in plan order keeps scheduling deterministic.
                for position in sorted(pending):
                    if len(futures) >= cap:
                        break
                    if not all(state.get(dep) == "completed" for dep in deps[position]):
                        continue
                    pending.discard(position)
                    futures[asyncio.ensure_future(run_one(tasks[position]))] = position

                if not futures:
                    # Nothing runnable and nothing in flight. Every task left is
                    # waiting on a dependency that did not complete, so waiting
                    # for them would be waiting forever; the skip pass below
                    # reports them instead.
                    break

                # NOTE: ``asyncio.wait`` hands back the *original* futures, whereas
                # ``as_completed`` yields the result values and would lose the
                # future -> task mapping (and re-raise a worker's exception, which
                # would abort the whole cycle). Collecting one failure at a time here
                # keeps a single bad task from taking the run down with it.
                finished, _ = await asyncio.wait(
                    set(futures), return_when=asyncio.FIRST_COMPLETED
                )
                for future in finished:
                    position = futures.pop(future)
                    task = tasks[position]
                    try:
                        result = future.result()
                    # Deliberately blind: any worker failure is captured as a
                    # task result. Narrowing this would let one bad task abort
                    # the whole cycle instead of failing in isolation.
                    except Exception as exc:  # noqa: BLE001
                        result = exc
                    state[position] = (
                        "failed" if isinstance(result, BaseException) else "completed"
                    )
                    done += 1
                    status.update(
                        f"Executed {done}/{total} task(s) "
                        f"(parallelism {max(1, self.settings.max_parallel_tasks)})"
                    )
                    if isinstance(result, BaseException):
                        task.status = "failed"
                        task.result = str(result)
                        console.print(f"[red]Task {task.id} failed: {result}[/red]")
                    else:
                        task.status = "completed"
                        task.result = str(result)
                        console.print(f"[green]Task {task.id} completed[/green]")
                    self._emit(
                        {
                            "type": "task_completed",
                            "id": task.id,
                            "status": task.status,
                            "result": str(task.result or "")[:_RESULT_CHARS],
                        }
                    )

        # Anything still pending lost a dependency race: upstream work failed, so
        # this task never gets to run and its own result would be a lie.
        for position in sorted(pending):
            unmet = sorted(
                {tasks[dep].id for dep in deps[position] if state.get(dep) != "completed"}
            )
            self._skip_task(
                tasks[position],
                "unmet dependency: "
                + (", ".join(unmet) or "dependency never ran")
                + " did not complete successfully",
            )

    def _skip_task(self, task: TaskModel, reason: str) -> None:
        """Mark a task as never-run and report it.

        Skipped tasks are still reported: the user asked for a dependency graph
        and is owed the truth about which parts of the plan did not happen.
        """
        task.status = SKIPPED
        task.result = reason
        console.print(f"[yellow]Task {task.id} {SKIPPED}: {reason}[/yellow]")
        self._emit(
            {
                "type": "task_completed",
                "id": task.id,
                "status": task.status,
                "result": str(task.result or "")[:_RESULT_CHARS],
            }
        )

    def _dependency_warning(
        self, reason: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        """Surface a plan's dependency defects without treating them as fatal."""
        console.print(f"[yellow]Dependency {reason}: {message}[/yellow]")
        self._emit(
            {
                "type": "dependency_warning",
                "reason": reason,
                "message": message,
                **(details or {}),
            }
        )

    def planner(self) -> PlannerAgent:
        return PlannerAgent(client=self.client, settings=self.settings)

    def worker(self, task: TaskModel) -> WorkerAgent:
        return WorkerAgent(task=task, client=self.client, settings=self.settings)

    def tester(self) -> TesterAgent:
        return TesterAgent(client=self.client, settings=self.settings)

    def _merge_tasks(self, new_tasks: list[TaskModel]) -> None:
        """Fold this cycle's tasks into ``self.tasks``.

        Tasks are keyed by ``id`` so re-planning a task replaces the previous
        version (keeping the newest status/result) instead of duplicating it.
        """
        index = {task.id: pos for pos, task in enumerate(self.tasks)}
        for task in new_tasks:
            if task.id in index:
                self.tasks[index[task.id]] = task
            else:
                index[task.id] = len(self.tasks)
                self.tasks.append(task)

    def _focus(self) -> str:
        if self.consecutive_failures > 0:
            return f"fixing {self.consecutive_failures} consecutive failing cycle(s)"
        if not self.tasks:
            return "first slice of the original goal"
        return "advancing the original goal"

    def _is_done(self) -> bool:
        """True when there is genuinely nothing left to do.

        Note this is NOT ``completed >= len(tasks)``: that tautology compares a
        count against the number of *unplanned* tasks and evaluated ``0 >= 0`` on
        the old empty stub, which halted the loop after exactly one cycle.
        """
        if not self.tasks:
            return True
        return self.tasks_planned_last_cycle == 0

    def _get_tasks(self) -> list[TaskModel]:
        return self.tasks

    def _emit(self, event: dict[str, Any]) -> None:
        """Stamp an event and hand it to ``on_event``; observers cannot break us."""
        event["type"] = str(event.get("type", "event"))
        event["cycle"] = self.cycles
        hook = self.on_event
        if hook is None:
            return
        try:
            hook(event)
        # Deliberately blind: observers must not be able to break the run.
        except Exception:  # noqa: BLE001
            return

    def _halt(self, reason: str, style: str = "yellow") -> None:
        console.print(f"[bold {style}]{reason}. Halting.[/bold {style}]")
        self._emit(
            {
                "type": "halted",
                "reason": reason,
                "cycles": self.cycles,
                "consecutive_failures": self.consecutive_failures,
            }
        )