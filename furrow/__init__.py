"""Furrow: an autonomous coding agent that runs a bounded development loop.

Only :class:`LLMClient` and :class:`Settings` are imported eagerly. Everything
else resolves through :func:`__getattr__` (PEP 562) so that ``import furrow``
stays cheap and free of import cycles, while ``from furrow import Orchestrator``
keeps working unchanged.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from furrow.config import Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.agents import PlannerAgent, TesterAgent, WorkerAgent
    from furrow.core import Event, EventBus, Orchestrator
    from furrow.state import RunState, StateStore

__version__ = "0.2.0"

_LAZY_EXPORTS: dict[str, str] = {
    "Event": "furrow.core",
    "EventBus": "furrow.core",
    "Orchestrator": "furrow.core",
    "PlannerAgent": "furrow.agents",
    "WorkerAgent": "furrow.agents",
    "TesterAgent": "furrow.agents",
    "RunState": "furrow.state",
    "StateStore": "furrow.state",
}

__all__ = [
    "Event",
    "EventBus",
    "LLMClient",
    "Orchestrator",
    "PlannerAgent",
    "RunState",
    "Settings",
    "StateStore",
    "TesterAgent",
    "WorkerAgent",
    "__version__",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(__all__))