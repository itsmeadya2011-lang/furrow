"""Agent implementations for Furrow.

This package contains the three core agents that drive the Furrow development loop:

- **PlannerAgent**: Breaks high-level goals into parallelizable coding tasks
- **WorkerAgent**: Implements individual tasks using file operations and shell commands
- **TesterAgent**: Detects and runs the project's test suite to verify correctness

Each agent uses a dedicated LLM model (configurable via Settings) and communicates
through structured JSON prompts/responses.
"""

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent

__all__ = ["PlannerAgent", "TesterAgent", "WorkerAgent"]
