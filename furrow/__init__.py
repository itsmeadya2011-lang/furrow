from __future__ import annotations

__all__ = ["LLMClient", "Settings", "get_settings"]


def __getattr__(name: str) -> object:
    if name == "LLMClient":
        from furrow.llm import LLMClient
        return LLMClient
    if name == "Settings":
        from furrow.config import Settings
        return Settings
    if name == "get_settings":
        from furrow.config import get_settings
        return get_settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
