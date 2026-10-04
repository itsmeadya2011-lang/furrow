"""Shared fixtures for the furrow regression suite.

Three things live here, and all three exist because this suite has to pass on a developer
machine that has real credentials in its environment *and* on CI, which has none:

* **Hermetic environment** -- :func:`hermetic_env` (autouse) deletes ``ANTHROPIC_API_KEY``,
  ``OPENAI_API_KEY`` and every ``FURROW_*`` variable, and disables dotenv loading. Without
  the dotenv switch a developer's local ``.env`` silently changes every default the
  :mod:`furrow.config` tests assert. ``get_settings`` is ``lru_cache``d, so the cache is
  rebuilt with ``refresh=True`` afterwards; skip that and one test that sets ``FURROW_*``
  changes what every later test in the run sees.
* **Settings and a temp workspace** -- ``settings`` / ``tmp_workspace``. Agent tests must
  never read or write the checkout, and ``ensure_api_key()`` must never be able to raise.
* **:class:`FakeLLM` / ``fake_llm``** -- a drop-in ``LLMClient`` that never touches a
  network, with the filesystem helpers left real.

:func:`pytest_configure` additionally silences pytest's ``PytestCollectionWarning`` about
:class:`furrow.models.TestResult`; see the comment there for why it is done that way.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from furrow.config import Settings, get_settings
from furrow.llm import LLMClient, LLMError

#: Credential variables a developer machine is likely to have exported.
CREDENTIAL_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

#: Every ``FURROW_*`` variable is config for this application and none of it should survive
#: into a test; the hostile-env verification run sets a couple on purpose.
FURROW_ENV_PREFIX = "FURROW_"

#: Which of the two :func:`pytest_configure` strategies was used, recorded for the test run
#: report (see :func:`test_test_result_is_marked_uncollectable` in ``test_models.py``).
COLLECTION_WARNING_FIX = "not run"


def pytest_configure(config: pytest.Config) -> None:
    """Stop pytest trying to collect ``furrow.models.TestResult`` as a test class.

    ``TestResult`` is a domain model whose name happens to start with ``Test``, so any test
    module that imports it (they all do -- the tester agent returns one) makes pytest emit
    ``PytestCollectionWarning: cannot collect test class 'TestResult' because it has a
    __init__ constructor``. Setting ``__test__ = False`` on the class is pytest's supported
    opt-out, and pydantic v2 models accept new *class* attributes even though they reject
    new instance attributes. So try that first and keep the output genuinely clean; fall
    back to a ``filterwarnings`` line, which only hides the symptom, if the attribute ever
    stops being settable. ``pyproject.toml`` is not ours to edit.
    """
    global COLLECTION_WARNING_FIX  # noqa: PLW0603  (one-shot module-level record)

    from furrow.models import TestResult

    try:
        TestResult.__test__ = False  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        config.addinivalue_line("filterwarnings", "ignore::pytest.PytestCollectionWarning")
        COLLECTION_WARNING_FIX = "filterwarnings"
    else:
        COLLECTION_WARNING_FIX = "__test__"


@pytest.fixture(scope="session", autouse=True)
def _settings_cache_boundary() -> Iterator[None]:
    """Rebuild the cached settings singleton at both ends of the session.

    Session-scoped fixtures are finalised *after* every function-scoped one, so the teardown
    below runs once ``monkeypatch`` has already restored the environment. That matters:
    ``get_settings`` is ``lru_cache``d, and a stale entry holding a test's key or
    ``FURROW_*`` override would otherwise outlive the run and leak into the next one.
    """
    get_settings(refresh=True)
    yield
    get_settings(refresh=True)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ambient credentials and configuration so no test can depend on them.

    Three sources have to be closed off, not one:

    1. ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` from the developer's shell,
    2. any ``FURROW_*`` override (the hostile-env verification run sets two on purpose),
    3. ``Settings(env_file=".env")``, which is resolved against the process CWD, so a
       developer's ``.env`` would otherwise change the defaults the config tests assert.

    ``monkeypatch`` restores 1 and 2 for us; ``monkeypatch.setitem`` restores 3. The
    ``refresh=True`` rebuild is required because ``get_settings`` is ``lru_cache``d.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in list(os.environ):
        if name in CREDENTIAL_ENV_VARS or name.startswith(FURROW_ENV_PREFIX):
            monkeypatch.delenv(name, raising=False)
    get_settings(refresh=True)


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """An empty, per-test directory to stand in for a project root.

    Separate from ``tmp_path`` (and not autouse) so a test can keep fixtures that use
    ``tmp_path`` directly -- for instance one that writes a marker file next to the
    workspace -- without the two colliding.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def settings(tmp_workspace: Path) -> Settings:
    """Settings that can never fail an ``ensure_api_key()`` call.

    ``anthropic_api_key`` is set explicitly so that provider selection tests and agent
    tests do not depend on the ``FURROW_PROVIDER`` that the hostile-env run exports, and
    ``workspace`` points at a temp dir so nothing an agent writes lands in the checkout.
    """
    return Settings(workspace=tmp_workspace, anthropic_api_key="test-key")


class FakeLLM(LLMClient):
    """An :class:`~furrow.llm.LLMClient` whose only difference is where the text comes from.

    It **subclasses** ``LLMClient`` rather than using ``unittest.mock`` because the agents
    under test need more than a stubbed ``complete``: ``WorkerAgent`` genuinely writes
    files through ``client.write_file``, reads the existing tree through
    ``client.read_file`` / ``client.list_files``, and builds its prompt from those results.
    A mock would have to fake all three and would then be asserting on the fake. Only
    ``complete`` is overridden here, so the real filesystem helpers are exercised against a
    temp workspace -- which is exactly what a regression in them should be caught by.

    The provider SDK accessors are overridden to inert so a stray ``client.anthropic``
    cannot build a real client, raise ``ProviderAuthError`` for a missing key, or leave an
    unclosed connection pool behind.

    Construct it with a queue of canned strings, with a callable, or with ``raise_after`` to
    fail on the Nth and later calls::

        FakeLLM(responses=['```json\\n{"tasks": []}\\n```'])
        FakeLLM(responder=lambda prompt, system, model: "fixed")
        FakeLLM(responses=["a"], raise_after=1)
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        responder: Callable[[str, str, str | None], str] | None = None,
        settings: Settings | None = None,
        raise_after: int | None = None,
        raise_with: BaseException | type[BaseException] | None = None,
    ) -> None:
        super().__init__(settings)
        if isinstance(responses, str):
            # `list("{}")` is `["{", "}"]`, which would hand the caller one character per
            # call and fail somewhere deep inside a parser. Say what was meant instead.
            raise TypeError(
                "FakeLLM(responses=...) takes a sequence of replies, not a single string; "
                'use responses=["..."] or the `fake_llm` fixture, which takes them varargs'
            )
        self.responses: list[str] = list(responses or ())
        self.responder = responder
        self.raise_after = raise_after
        self.raise_with = raise_with
        #: One entry per ``complete`` call, in order, with the exact arguments received.
        self.calls: list[dict[str, Any]] = []

    @property
    def call_count(self) -> int:
        """How many times :meth:`complete` has been called."""
        return len(self.calls)

    def last_call(self) -> dict[str, Any]:
        """The most recent :meth:`complete` call record; fails loudly if there was none."""
        if not self.calls:
            raise AssertionError("complete() was never called")
        return self.calls[-1]

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the next canned reply instead of calling a provider.

        Args:
            prompt: User message; recorded verbatim.
            system: System prompt; recorded verbatim.
            model: Requested model id; recorded verbatim.
            max_tokens: Requested budget; recorded verbatim.

        Returns:
            The next entry of ``responses``, or ``responder(...)`` when a callable was given.

        Raises:
            BaseException: ``raise_with`` once more than ``raise_after`` calls have been
                recorded. The call is still recorded first, so the failing call is visible
                in :attr:`calls` like any other.
            LLMError: No responses left and no responder -- an exhausted queue is a test
                bug, and saying so beats returning an empty completion that the caller
                would fail to parse for an unrelated-looking reason.
        """
        self.calls.append(
            {"prompt": prompt, "system": system, "model": model, "max_tokens": max_tokens}
        )
        if self.raise_after is not None and len(self.calls) > self.raise_after:
            raise self._error()
        if self.responder is not None:
            return self.responder(prompt, system, model)
        if not self.responses:
            raise LLMError(
                f"FakeLLM ran out of canned responses after {self.call_count} call(s)"
            )
        return self.responses.pop(0)

    def _error(self) -> BaseException:
        """Build a fresh exception instance from :attr:`raise_with`."""
        exc = self.raise_with
        if exc is None:
            return LLMError(f"FakeLLM was told to fail after {self.raise_after} calls")
        return exc() if isinstance(exc, type) else exc

    @property
    def anthropic(self) -> None:  # type: ignore[override]  # deliberately incompatible
        """Inert: never builds a real ``AsyncAnthropic``, so it can never raise for a key."""
        return None

    @property
    def openai(self) -> None:  # type: ignore[override]  # deliberately incompatible
        """Inert: never builds a real ``AsyncOpenAI``, so it can never raise for a key."""
        return None

    async def aclose(self) -> None:
        """Inert: there are no cached provider clients to close."""
        return None


@pytest.fixture
def fake_llm(settings: Settings) -> Callable[..., FakeLLM]:
    """Factory for a :class:`FakeLLM` bound to the hermetic ``settings`` fixture.

    Accepts the canned replies positionally and every :class:`FakeLLM` keyword::

        fake_llm('{"summary": "done", "files": []}')
        fake_llm(responder=lambda prompt, system, model: "{}")
        fake_llm("first", raise_after=1)

    Pass ``settings=`` explicitly to override the bound value (e.g. a different
    ``max_parallel_tasks`` or ``workspace``).
    """

    def build(*responses: str, **kwargs: Any) -> FakeLLM:
        kwargs.setdefault("settings", settings)
        return FakeLLM(responses, **kwargs)

    return build
