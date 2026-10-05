"""Furrow: an autonomous coding agent with an infinite parallel development loop."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.1.1"

__all__ = [
    "LLMClient",
    "Settings",
    "Orchestrator",
    "PlannerAgent",
    "WorkerAgent",
    "TesterAgent",
    "__version__",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "LLMClient": ("furrow.llm", "LLMClient"),
    "Settings": ("furrow.config", "Settings"),
    "Orchestrator": ("furrow.core.orchestrator", "Orchestrator"),
    "PlannerAgent": ("furrow.agents.planner", "PlannerAgent"),
    "WorkerAgent": ("furrow.agents.worker", "WorkerAgent"),
    "TesterAgent": ("furrow.agents.tester", "TesterAgent"),
}

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from furrow.agents.planner import PlannerAgent
    from furrow.agents.tester import TesterAgent
    from furrow.agents.worker import WorkerAgent
    from furrow.config import Settings
    from furrow.core.orchestrator import Orchestrator
    from furrow.llm import LLMClient


def __getattr__(name: str) -> Any:
    """Resolve public API members on first access to keep imports cheap."""
    try:
        module_name, attr = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
