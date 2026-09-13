from __future__ import annotations

from typing import TYPE_CHECKING

from furrow.config import TestResult

if TYPE_CHECKING:
    from furrow.core.orchestrator import Orchestrator


class LoopResult:
    """Result of running the plan-execute-test loop.

    Attributes:
        cycles: Number of cycles executed.
        goal_completed: Whether the goal was completed.
        tasks_completed: Number of tasks that completed successfully.
        tasks_failed: Number of tasks that failed.
        test_result: The test result from the final cycle, if any.
    """

    cycles: int
    goal_completed: bool
    tasks_completed: int
    tasks_failed: int
    test_result: TestResult | None

    def __init__(
        self,
        cycles: int,
        goal_completed: bool,
        tasks_completed: int,
        tasks_failed: int,
        test_result: TestResult | None,
    ) -> None:
        self.cycles = cycles
        self.goal_completed = goal_completed
        self.tasks_completed = tasks_completed
        self.tasks_failed = tasks_failed
        self.test_result = test_result

    def __repr__(self) -> str:
        return (
            f"LoopResult(cycles={self.cycles}, goal_completed={self.goal_completed}, "
            f"tasks_completed={self.tasks_completed}, tasks_failed={self.tasks_failed}, "
            f"test_result={self.test_result})"
        )


async def run_loop(goal: str, orchestrator: Orchestrator, max_cycles: int = 0) -> LoopResult:
    """Run the core plan-execute-test loop.

    Repeatedly cycles through planning, executing, and testing phases until
    the goal is complete or the maximum number of cycles is reached.

    Args:
        goal: The goal string for the agent to pursue.
        orchestrator: The Orchestrator instance that manages the cycle logic.
        max_cycles: Maximum number of cycles to run. A value of 0 means
            unlimited cycles until the goal is complete.

    Returns:
        A LoopResult summarizing the loop execution including cycle count,
        task status, and test result.
    """
    cycles = 0
    test_result: TestResult | None = None

    while max_cycles == 0 or cycles < max_cycles:
        cycles += 1
        await orchestrator._cycle()
        if orchestrator._is_done():
            break

    tasks_completed = sum(1 for t in orchestrator._get_tasks() if t.status == "completed")
    tasks_failed = sum(1 for t in orchestrator._get_tasks() if t.status == "failed")
    goal_completed = orchestrator._is_done()

    return LoopResult(
        cycles=cycles,
        goal_completed=goal_completed,
        tasks_completed=tasks_completed,
        tasks_failed=tasks_failed,
        test_result=test_result,
    )
