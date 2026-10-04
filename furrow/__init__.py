"""Furrow — autonomous coding agent with an infinite parallel development loop."""

# LLMClient is deliberately NOT re-exported here. Importing it eagerly would make
# `import furrow` pull in the anthropic/openai SDKs through furrow.llm, which is slow and
# makes the whole package unimportable without those optional deps installed.
# Callers should import it directly: `from furrow.llm import LLMClient`.
from furrow.config import Provider, Settings, get_settings, settings
from furrow.models import (
    FileChange,
    Plan,
    TaskModel,
    TaskStatus,
    TestResult,
    WorkerResult,
)

__version__ = "0.1.0"  # keep in sync with the version in pyproject.toml

__all__ = [
    "FileChange",
    "Plan",
    "Provider",
    "Settings",
    "TaskModel",
    "TaskStatus",
    "TestResult",
    "WorkerResult",
    "__version__",
    "get_settings",
    "settings",
]
