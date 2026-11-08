"""Furrow - autonomous coding agent with an infinite parallel development loop."""

from importlib.metadata import PackageNotFoundError, version

from furrow.config import Settings
from furrow.llm import LLMClient

try:
    __version__ = version("furrow")
except PackageNotFoundError:  # pragma: no cover - package not installed
    __version__ = "0.0.0"

__all__ = ["LLMClient", "Settings", "__version__"]
