from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from furrow.config import Settings
from furrow.llm import LLMClient

try:
    __version__ = version("furrow")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["LLMClient", "Settings", "__version__"]
