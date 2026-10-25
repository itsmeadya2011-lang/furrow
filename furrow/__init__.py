"""Furrow - autonomous coding agent with an infinite parallel development loop."""

from __future__ import annotations

import importlib.metadata

from furrow.config import Settings
from furrow.llm import LLMClient

try:
    __version__ = importlib.metadata.version("furrow")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover - installed in dev
    __version__ = "0.0.0"

__all__ = ["LLMClient", "Settings", "__version__"]
