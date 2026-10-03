"""The package surface must stay cheap: ``import furrow`` pulls in no optional dependency.

``furrow/__init__.py`` re-exports :class:`~furrow.llm.LLMClient` and
:class:`~furrow.config.Settings` through PEP 562 ``__getattr__``. These tests pin
that laziness in a *fresh interpreter* (an in-process check cannot work: the rest
of the suite imports ``furrow.llm`` and ``furrow.web`` long before this file
runs) and that the re-exports keep behaving like ordinary module attributes.

No network, no credentials, no real provider calls: the dependency-missing check
simulates an incomplete environment with an import hook instead of uninstalling
anything.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import furrow
from furrow import LLMClient, Settings
from furrow.config import Settings as ConfigSettings
from furrow.llm import LLMClient as ModuleLLMClient

_ROOT = str(Path(furrow.__file__).resolve().parent.parent)

# Modules ``import furrow`` must not drag in, directly or transitively.
_HEAVY = (
    "furrow.llm",
    "furrow.web",
    "anthropic",
    "openai",
    "aiofiles",
    "uvicorn",
    "fastapi",
)

# What the "optional dependency is not installed" simulation hides.
_OPTIONAL = (
    "anthropic",
    "openai",
    "aiofiles",
    "uvicorn",
    "fastapi",
    "httpx",
    "tenacity",
    "structlog",
    "click",
    "rich",
)

_FRESH_LAZY = """
import sys
import furrow

pulled_in = [name for name in {heavy!r} if name in sys.modules]
assert not pulled_in, pulled_in

# Resolving a re-export must not drag in the rest of the chain either.
from furrow import Settings

assert Settings.__name__ == "Settings"
pulled_in = [name for name in {heavy!r} if name in sys.modules]
assert not pulled_in, pulled_in
print("ok")
"""

_FRESH_NO_OPTIONAL_DEPS = """
import sys

BLOCKED = {optional!r}


class _Blocker:
    \"\"\"Meta-path finder that makes the optional dependencies look uninstalled.\"\"\"

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{root!r}}", name=root)
        return None


sys.meta_path.insert(0, _Blocker())

try:
    import anthropic
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("the dependency blocker never took effect")

import furrow

from furrow import Settings

assert Settings.__name__ == "Settings"
print("ok")
"""


def _run_fresh(script: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` in a new interpreter that can import this checkout and nothing else."""
    env = dict(os.environ, PYTHONPATH=_ROOT, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )


def test_public_names_resolve_to_the_defining_modules():
    """``from furrow import X`` still hands back the very class the module defines."""
    assert furrow.__all__ == ["LLMClient", "Settings"]
    assert LLMClient is ModuleLLMClient
    assert Settings is ConfigSettings
    assert getattr(furrow, "LLMClient") is LLMClient
    assert getattr(furrow, "Settings") is Settings


def test_dir_lists_the_public_surface():
    listed = dir(furrow)
    for name in furrow.__all__:
        assert name in listed


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'no_such_export'"):
        getattr(furrow, "no_such_export")
    assert "no_such_export" not in dir(furrow)


def test_import_furrow_is_lazy_in_a_fresh_interpreter():
    result = _run_fresh(_FRESH_LAZY.format(heavy=_HEAVY))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_import_furrow_works_without_the_optional_dependencies():
    result = _run_fresh(_FRESH_NO_OPTIONAL_DEPS.format(optional=_OPTIONAL))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"