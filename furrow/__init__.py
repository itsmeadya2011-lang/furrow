"""Furrow - autonomous coding agent with an infinite parallel development loop."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from furrow.config import Settings
from furrow.llm import LLMClient

try:
    __version__ = _pkg_version("furrow")
except PackageNotFoundError:  # pragma: no cover - running from source without install
    __version__ = "0.1.0"

__all__ = ["LLMClient", "Settings", "__version__"]
