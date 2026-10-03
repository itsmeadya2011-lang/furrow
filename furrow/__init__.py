"""Furrow's public surface, resolved lazily (PEP 562).

Re-exporting the client and the settings eagerly would make ``import furrow``
execute :mod:`furrow.llm` (anthropic, openai, aiofiles) and :mod:`furrow.web`
(uvicorn, fastapi), so the cheapest possible import would fail whenever one of
those optional dependencies is missing -- and would pay for the whole chain even
when it does not.  ``__getattr__`` below imports each name only on first lookup
and caches it in the module namespace, so ``import furrow`` stays dependency-free
while ``from furrow import LLMClient`` keeps working exactly as it did before.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-free at runtime, exact for type checkers
    from furrow.config import Settings
    from furrow.llm import LLMClient

__all__ = ["LLMClient", "Settings"]

# Public name -> module that defines it, consulted by ``__getattr__``.
_EXPORTS = {"LLMClient": "furrow.llm", "Settings": "furrow.config"}


def __getattr__(name: str) -> Any:
    """Import a re-exported name on demand (PEP 562)."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), name)
    globals()[name] = value  # cache so later lookups skip __getattr__ entirely
    return value


def __dir__() -> list[str]:
    """List the public surface even before anything has been imported."""
    return sorted(set(globals()) | set(__all__))