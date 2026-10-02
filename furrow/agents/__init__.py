"""Agent subpackage.

Agent classes are re-exported lazily (:pep:`562` ``__getattr__``) because eager
imports here close an import cycle: ``furrow.core.orchestrator`` imports these
agents, while the agents ultimately import ``furrow.llm``, which is itself
imported by ``furrow/__init__.py``.  Importing them eagerly re-enters a
half-initialised ``furrow.llm`` and raises ``ImportError``.  Deferring to first
attribute access breaks the cycle and keeps the public surface unchanged.  Do
not revert to eager imports.
"""

from __future__ import annotations

from typing import Any

__all__ = ["PlannerAgent", "TesterAgent", "WorkerAgent"]


def __getattr__(name: str) -> Any:
    if name == "PlannerAgent":
        from furrow.agents.planner import PlannerAgent

        return PlannerAgent
    if name == "WorkerAgent":
        from furrow.agents.worker import WorkerAgent

        return WorkerAgent
    if name == "TesterAgent":
        from furrow.agents.tester import TesterAgent

        return TesterAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
