"""The three agents that make up a furrow run.

* :class:`~furrow.agents.planner.PlannerAgent` turns a goal into a validated plan.
* :class:`~furrow.agents.worker.WorkerAgent` executes one task with a bounded tool loop.
* :class:`~furrow.agents.tester.TesterAgent` runs the workspace's tests and reports.

Importing this package does not build an LLM client, read settings, or touch the
network: agents construct their client lazily in ``__init__``.
"""

from __future__ import annotations

from furrow.agents.planner import PlannerAgent, PlannerError
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent

__all__ = ["PlannerAgent", "PlannerError", "TesterAgent", "WorkerAgent"]
