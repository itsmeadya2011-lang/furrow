"""Package entry point.

The re-exports below are deliberately **lazy** (PEP 562 module-level
``__getattr__``).  ``furrow.config`` ends with ``settings = Settings()``, so
importing it eagerly at package-import time makes a malformed ``.env`` or
``FURROW_*`` value raise a raw pydantic ``ValidationError`` out of ``import
furrow`` itself - before any of the friendly error handling in
:mod:`furrow.cli.main` gets a chance to run.  Resolving the names on first
access keeps ``from furrow import Settings`` and ``from furrow import
LLMClient`` working for every existing caller while leaving ``import furrow``
free of side effects, so a bad environment surfaces as a message and a non-zero
exit code from the CLI instead of an import-time traceback.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - satisfied for type checkers, skipped at runtime
    from furrow.config import Settings
    from furrow.llm import LLMClient

__all__ = ["LLMClient", "Settings"]

#: Public name -> (module that defines it, attribute name).  ``furrow.llm`` imports
#: ``furrow.config``, so *both* names have to be lazy for ``import furrow`` to
#: avoid building the settings singleton.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "LLMClient": ("furrow.llm", "LLMClient"),
    "Settings": ("furrow.config", "Settings"),
}


def __getattr__(name: str) -> Any:
    """Resolve ``LLMClient`` / ``Settings`` on first access, then cache them."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value  # cache, so subsequent lookups bypass __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)