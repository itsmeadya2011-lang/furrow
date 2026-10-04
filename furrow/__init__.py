"""Furrow: an autonomous coding agent for parallel development."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from furrow.config import (
    LOG_LEVELS,
    ConfigurationError,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    configure_logging,
    get_settings,
    normalize_log_level,
)
from furrow.llm import LLMClient

__all__ = [
    "LLMClient",
    "Settings",
    "Provider",
    "TaskStatus",
    "TaskModel",
    "Plan",
    "TestResult",
    "ConfigurationError",
    "get_settings",
    "configure_logging",
    "normalize_log_level",
    "LOG_LEVELS",
    "__version__",
]

try:
    __version__ = version("furrow")
except PackageNotFoundError:
    __version__ = "0.1.0"
