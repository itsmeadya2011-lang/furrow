"""Shared pytest fixtures and the offline ``FakeLLM`` test double.

Test modules code against the names defined here, so these names and signatures
are part of the suite's public contract:

* :class:`FakeLLM` -- a scripted, network-free stand-in for ``furrow.llm.LLMClient``.
* ``tmp_workspace`` / ``make_settings`` / ``write_file`` -- workspace + settings helpers.
* ``reset_settings_singleton`` -- autouse guard around the mutable
  ``furrow.config.settings`` singleton.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from furrow import config as config_module
from furrow.config import Settings
from furrow.llm import LLMClient

#: Raised by :meth:`FakeLLM.complete` when the script is empty and no default exists.
EXHAUSTED_MESSAGE = "FakeLLM ran out of scripted responses"


@contextmanager
def _no_furrow_env() -> Iterator[None]:
    """Temporarily hide every ``FURROW_*`` environment variable.

    ``_env_file=None`` already stops pydantic-settings from reading ``.env``, but the
    environment is still consulted. Clearing the prefixed variables keeps a stray
    ``FURROW_MAX_PARALLEL_TASKS`` in the developer's shell from leaking into a test.
    """
    saved = {key: value for key, value in os.environ.items() if key.startswith("FURROW_")}
    for key in saved:
        del os.environ[key]
    try:
        yield
    finally:
        for key in [k for k in os.environ if k.startswith("FURROW_")]:
            del os.environ[key]
        os.environ.update(saved)


def make_settings_for(
    workspace: str | Path | None = None,
    **overrides: Any,
) -> Settings:
    """Build an isolated ``Settings`` for ``workspace``.

    No ``.env`` file, no ``FURROW_*`` environment variable and no ambient API key can
    influence the result; an explicit workspace and a dummy Anthropic key are always
    supplied. ``workspace=None`` gets a fresh private temp directory.
    """
    path = Path(workspace) if workspace is not None else Path(tempfile.mkdtemp(prefix="furrow-ws-"))
    path.mkdir(parents=True, exist_ok=True)
    with _no_furrow_env():
        return Settings(_env_file=None, workspace=path, anthropic_api_key="test-key", **overrides)


class FakeLLM(LLMClient):
    """Scripted ``LLMClient`` double. Never performs any network call.

    ``responses`` is consumed in order by successive :meth:`complete` calls; an element
    that is an ``Exception`` *instance* is raised instead of returned. When the script
    runs out, ``default`` is returned when given, otherwise an ``AssertionError`` is
    raised so a mismatched test fails loudly instead of comparing against ``""``.
    """

    def __init__(
        self,
        responses: list[str | Exception] | None = None,
        *,
        default: str | None = None,
        workspace: str | Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        # Keep the real LLMClient state (``self.settings`` is always a real Settings).
        if settings is not None:
            resolved = settings
        else:
            resolved = make_settings_for(workspace)
        super().__init__(settings=resolved)
        self._responses: list[str | Exception] = list(responses or [])
        self._default = default
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.models: list[str | None] = []
        self.calls: int = 0

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.systems.append(system)
        self.models.append(model)
        if not self._responses:
            if self._default is not None:
                return self._default
            raise AssertionError(EXHAUSTED_MESSAGE)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def read_file(self, path: str | Path) -> str:
        return Path(path).read_text(encoding="utf-8")

    async def write_file(self, path: str | Path, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def list_files(self, directory: str | Path) -> list[str]:
        root = Path(directory)
        if not root.exists():
            return []
        return [str(f.relative_to(root)) for f in root.rglob("*") if f.is_file()]


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """An empty directory serving as the workspace root for one test."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def make_settings() -> Callable[..., Settings]:
    """Factory building an isolated ``Settings``; use as ``make_settings(tmp_workspace)``."""

    def _make(workspace: str | Path | None = None, **overrides: Any) -> Settings:
        return make_settings_for(workspace, **overrides)

    return _make


@pytest.fixture
def write_file(tmp_workspace: Path) -> Callable[..., Path]:
    """Write text into the workspace and return the written ``Path``.

    A relative path is resolved against ``tmp_workspace``; an absolute path is used as
    given, which also allows writing into an unrelated temporary directory.
    """

    def _write(
        path: str | Path,
        content: str = "",
        *,
        workspace: str | Path | None = None,
    ) -> Path:
        root = Path(workspace) if workspace is not None else tmp_workspace
        target = Path(path)
        if not target.is_absolute():
            target = root / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    return _write


@pytest.fixture(autouse=True)
def reset_settings_singleton() -> Iterator[Settings]:
    """Restore ``furrow.config.settings`` after every test.

    The CLI imports the singleton by value and mutates it in place (provider,
    workspace, per-role models, log level), so without this guard those changes leak
    into every test that runs afterwards. Field values are restored on the original
    object -- restoring attributes rather than replacing it keeps the copy already
    imported by ``furrow.cli.main`` in sync -- and the module attribute is rebound in
    case a test swapped it out.
    """
    original = config_module.settings
    saved = original.model_dump()

    yield original

    for name, value in saved.items():
        setattr(original, name, value)
    config_module.settings = original
