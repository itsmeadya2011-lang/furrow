"""Core orchestration and state management for Furrow.

This package contains the central components that coordinate the Furrow development loop:

- **Orchestrator**: The main coordinator that runs planning, execution, and testing cycles
- **EventBus**: Pub/sub system for streaming real-time events to CLI and web UI
- **StateStore**: Atomic JSON persistence for crash recovery and resumption
- **RunState**: Complete run state including tasks, history, and usage metrics
- **runner**: Shared utilities for running the orchestrator (used by CLI and web)

The orchestrator manages the infinite loop:
1. Creates a plan via PlannerAgent
2. Executes tasks in dependency-ordered waves via WorkerAgent
3. Runs tests via TesterAgent
4. Updates goal with test feedback and repeats

State is persisted after each cycle to enable recovery from interruption.
"""

from furrow.core.events import Event, EventBus, EventType
from furrow.core.orchestrator import Orchestrator
from furrow.core.runner import (
    build_run_settings,
    create_orchestrator,
    create_state_store,
    release_client,
    run_orchestrator,
)
from furrow.core.state import RunState, StateStore

__all__ = [
    "Event",
    "EventBus",
    "EventType",
    "Orchestrator",
    "RunState",
    "StateStore",
    "build_run_settings",
    "create_orchestrator",
    "create_state_store",
    "release_client",
    "run_orchestrator",
]
