"""Agent implementations that drive the Furrow plan/execute/test loop.

Each agent owns one stage and is constructed independently so it can be driven
straight from a test or a script:

- :class:`PlannerAgent` turns a goal into a validated plan.
- :class:`WorkerAgent` implements one task, writing the changed files to disk.
- :class:`TesterAgent` runs the suite and judges the outcome.

Example:
    ```python
    from furrow.agents import PlannerAgent, TesterAgent, WorkerAgent
    ```
"""

from furrow.agents.parsing import (
    coerce_status,
    extract_json,
    parse_json_response,
    strip_code_fence,
)
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import CommandResult, TesterAgent
from furrow.agents.worker import WorkerAgent

__all__ = [
    "CommandResult",
    "PlannerAgent",
    "TesterAgent",
    "WorkerAgent",
    "coerce_status",
    "extract_json",
    "parse_json_response",
    "strip_code_fence",
]
