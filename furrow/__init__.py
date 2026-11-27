"""Furrow - autonomous coding agent with an infinite parallel development loop."""

from importlib.metadata import version as _distribution_version

from furrow.config import Settings
from furrow.llm import LLMClient

try:
    __version__ = _distribution_version("furrow")
except Exception:  # pragma: no cover - installed via pip/setuptools normally
    __version__ = "0.0.0"

__all__ = ["LLMClient", "Settings", "__version__"]
