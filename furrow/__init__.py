"""Furrow - autonomous coding agent with an infinite parallel development loop."""

__version__ = "0.1.0"

from furrow.config import Settings
from furrow.llm import LLMClient

__all__ = ["LLMClient", "Settings"]
