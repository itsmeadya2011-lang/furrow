"""Furrow — an autonomous coding agent built around a bounded development loop.

The loop plans a goal into parallel tasks, executes them with tool-using worker
agents, runs the project's tests, and stops on a cycle budget, an empty plan, a
stall threshold, or an explicit stop request. See ``core/orchestrator.py``.

The public surface is re-exported here so callers only need ``import furrow``.
"""

from furrow.config import Plan, Provider, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.events import Event, EventBus, EventType
from furrow.core.orchestrator import Orchestrator
from furrow.core.reporting import RichReporter
from furrow.llm import LLMClient
from furrow.tools import WorkspaceTools

__version__ = "0.1.0"

__all__ = [
    "Event",
    "EventBus",
    "EventType",
    "LLMClient",
    "Orchestrator",
    "Plan",
    "Provider",
    "RichReporter",
    "Settings",
    "TaskModel",
    "TaskStatus",
    "TestResult",
    "WorkspaceTools",
    "__version__",
]
