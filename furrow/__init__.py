"""Furrow - autonomous coding agent with an infinite parallel development loop."""

from furrow.config import Settings
from furrow.llm import LLMClient

__version__ = "0.1.0"

__all__ = ["LLMClient", "Settings", "__version__"]
