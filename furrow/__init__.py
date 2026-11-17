"""Furrow - Autonomous coding agent with an infinite parallel development loop.

Furrow is an autonomous coding agent that breaks down high-level goals into
parallelizable tasks, executes them using LLM-powered workers, and continuously
verifies the results by running tests. It operates in cycles:

1. **Plan**: The planner agent breaks the goal into independent tasks
2. **Execute**: Worker agents implement tasks in parallel using tools
3. **Test**: The tester agent runs the test suite to verify correctness
4. **Repeat**: Failed tests generate feedback for the next cycle

Key features:
- Parallel task execution with dependency resolution
- Automatic test detection and execution (pytest, npm test, cargo test, go test)
- State persistence for crash recovery and resumption
- Web UI and CLI interfaces
- Multi-provider LLM support (Anthropic, OpenAI, Ollama)
- Cost tracking and token usage monitoring
- Security sandboxing for command execution

Example:
    >>> from furrow import Settings
    >>> from furrow.core.orchestrator import Orchestrator
    >>> settings = Settings(model="claude-sonnet-4-20250514")
    >>> orchestrator = Orchestrator("Build a REST API with FastAPI", settings=settings)
    >>> result = await orchestrator.run()
"""

from furrow.config import Settings
from furrow.llm import LLMClient

__all__ = ["LLMClient", "Settings"]
__version__ = "0.1.0"
