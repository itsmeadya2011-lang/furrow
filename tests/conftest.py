"""Shared fakes and fixtures for the Furrow test suite.

Three rules hold for every test in this suite:

* **No provider traffic.** Every LLM interaction goes through :class:`FakeLLMClient`, so no
  test may depend on ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` or on the network.
* **No real waiting.** The tester's subprocess budgets are clamped by an autouse fixture, so
  a test that provokes a timeout costs about two seconds instead of minutes.
* **No vacuous passes.** Every scripted double raises :class:`AssertionError` when it runs out
  of scripted behaviour, so a runaway loop fails the test instead of quietly succeeding.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TestResult
from furrow.llm import LLMClient

# --------------------------------------------------------------------------------------
# pytest collection
# --------------------------------------------------------------------------------------
# WHY these lines are here: pytest collects *any* class whose name starts with ``Test``
# that it can see in a test module -- including classes imported there, however unrelated
# they are. So the rule for this suite is general: **every** imported class whose name
# starts with ``Test`` must be marked with ``__test__ = False``, which is pytest's
# documented opt-out and is applied once here, centrally, so no test module has to carry
# the ceremony and no run prints a spurious warning. Two such classes reach this suite:
#
# * ``TestResult`` -- a plain pydantic model in ``furrow.config``, imported by nearly every
#   test, which produced ``PytestCollectionWarning: cannot collect test class 'TestResult'``
#   on every run, and
# * ``TesterAgent`` -- the real agent from ``furrow.agents.tester``, imported by the agent
#   tests, which triggers the identical warning.
#
# The lines are not arbitrary and must not be removed.
TestResult.__test__ = False
TesterAgent.__test__ = False

#: Set by the test that drives the tester agent against a *real* ``pytest`` subprocess.
#:
#: The agent is supposed to run that subprocess in ``settings.workspace``. If it ignored the
#: workspace and inherited the test process's directory, the nested ``pytest`` would collect
#: this very suite again and recurse without end. A non-empty value makes collection in this
#: project fail fast instead -- see :func:`pytest_collection` -- so the bug surfaces as one
#: failing assertion rather than as a hung suite.
NESTED_RUN_GUARD = "FURROW_TEST_NESTED_RUN_GUARD"


def pytest_collection(session: pytest.Session) -> None:
    """Refuse to collect the suite from inside one of its own tests."""
    if os.environ.get(NESTED_RUN_GUARD):
        raise pytest.UsageError(
            f"{NESTED_RUN_GUARD} is set: refusing to collect this suite from inside a test "
            "(a runner subprocess inherited the test process's directory)"
        )


# --------------------------------------------------------------------------------------
# isolation
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _ignore_ambient_furrow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's exported ``FURROW_*`` variables out of the suite.

    ``Settings`` reads the environment with the ``FURROW_`` prefix, so an exported
    ``FURROW_MAX_CYCLES`` or ``FURROW_WORKSPACE`` would otherwise change loop behaviour.
    """
    for name in [name for name in os.environ if name.startswith("FURROW_")]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def bounded_runner_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clamp the tester agent's real subprocess budgets down to a couple of seconds.

    ``TEST_RUN_TIMEOUT_SECONDS`` is 120 and ``REAP_TIMEOUT_SECONDS`` is 10 in production, so
    without this a test that provokes either would stall the whole suite.
    """
    monkeypatch.setattr(tester_module, "TEST_RUN_TIMEOUT_SECONDS", 2.0)
    monkeypatch.setattr(tester_module, "REAP_TIMEOUT_SECONDS", 2.0)


# --------------------------------------------------------------------------------------
# the LLM double
# --------------------------------------------------------------------------------------


class FakeLLMClient(LLMClient):
    """An :class:`~furrow.llm.LLMClient` whose only stubbed method is ``complete()``.

    ``read_file``, ``write_file`` and ``list_files`` are inherited unchanged, so agent tests
    exercise the real filesystem behaviour while the provider is scripted.

    Args:
        responses: Model replies, handed out in order.
        settings: Passed straight to :class:`~furrow.llm.LLMClient`.
        repeat: When set, the final reply is repeated forever instead of being consumed.
    """

    def __init__(
        self,
        responses: list[str] | tuple[str, ...] = (),
        *,
        settings: Settings | None = None,
        repeat: bool = False,
    ) -> None:
        super().__init__(settings=settings)
        self.responses = list(responses)
        self.repeat = repeat
        #: Every call, in order, so a test can assert on prompts and models.
        self.calls: list[dict[str, Any]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        if not self.responses:
            raise AssertionError(
                f"FakeLLMClient.complete() was called {len(self.calls)} time(s) with no scripted "
                "reply left: the code under test looped or made an unplanned provider call"
            )
        if len(self.responses) == 1 and not self.repeat:
            return self.responses[0]
        return self.responses.pop(0)

    @property
    def prompts(self) -> list[str]:
        """Just the prompts, in call order."""
        return [call["prompt"] for call in self.calls]


@pytest.fixture
def fake_llm():
    """Return a factory for :class:`FakeLLMClient` instances."""

    def _make(responses: list[str] | tuple[str, ...] = (), **kwargs: Any) -> FakeLLMClient:
        return FakeLLMClient(responses, **kwargs)

    return _make