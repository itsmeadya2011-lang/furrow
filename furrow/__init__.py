from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "LLMClient": "furrow.llm",
    "Settings": "furrow.config",
}

__all__ = ["__version__", "LLMClient", "Settings"]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from furrow.config import Settings
    from furrow.llm import LLMClient
