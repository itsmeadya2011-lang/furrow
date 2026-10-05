"""Shared test doubles for the furrow suite.

Two names exported here are a hard contract for every test module:

* :class:`FakeLLM` - a stand-in for :class:`furrow.llm.LLMClient` that satisfies the
  :class:`furrow.agents.base.SupportsComplete` protocol, so an agent can be built
  without a network call or an API key.
* the ``settings`` fixture - a :class:`furrow.config.Settings` built from explicit
  keywords.  It deliberately does *not* hand back the module-level
  ``furrow.config.settings`` singleton: that object is constructed at import time
  from the developer's ``.env``, so any test reading it would silently pick up
  whatever credentials and limits happen to sit on the machine.  Because init
  keywords outrank both environment variables and the dotenv file in
  pydantic-settings, a fully explicit fixture cannot be redirected by either.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from furrow.config import Provider, Settings

__all__ = ["FakeLLM", "settings"]


class FakeLLM:
    """An injectable LLM client that replays canned responses.

    Exactly one of the response sources is used, in this precedence order:

    ``error``
        An exception instance raised by every :meth:`complete` call.  The call is
        still recorded in :attr:`calls` first, so a test can assert on what was
        asked before the failure.
    ``responses``
        A sequence consumed in order.  Once it is exhausted the final element is
        repeated, so a single-element list behaves like ``response`` and a
        multi-element one scripts a conversation.
    ``response``
        A single canned string returned by every call.

    Passing neither ``response`` nor ``responses`` raises :class:`TypeError`: a fake
    that silently returns empty output hides the test's real mistake.
    """

    def __init__(
        self,
        response: str | None = None,
        responses: Sequence[str] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.error = error
        if responses is not None:
            if not responses:
                raise TypeError("FakeLLM(responses=...) needs at least one response")
            self._responses: list[str] = list(responses)
        elif response is not None:
            self._responses = [response]
        elif error is None:
            raise TypeError("FakeLLM needs a response= or responses= value to replay")
        else:
            self._responses = []
        self.calls: list[tuple[str, str, str | None]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        """Record the call and return the next canned response."""
        self.calls.append((prompt, system, model))
        if self.error is not None:
            raise self.error
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return self._responses[index]

    def __repr__(self) -> str:
        return f"FakeLLM(responses={self._responses!r}, error={self.error!r})"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A fresh :class:`Settings` with a scratch workspace and small budgets.

    Every field is passed explicitly.  ``workspace`` points at ``tmp_path`` so a
    test that writes through the agents cannot touch the checkout, and the retry /
    timeout / token budgets are tiny so a misbehaving fake fails fast instead of
    stalling the suite.  The host/port values are deliberately non-default so a
    test that accidentally binds the real defaults is obvious in its assertions.
    """
    configured = Settings(
        provider=Provider.ANTHROPIC,
        model="test-model",
        planner_model="test-planner-model",
        worker_model="test-worker-model",
        tester_model="test-tester-model",
        anthropic_api_key=None,
        openai_api_key=None,
        ollama_base_url="http://localhost:11434",
        ollama_model="test-ollama-model",
        max_parallel_tasks=2,
        max_cycles=3,
        max_consecutive_failures=2,
        max_tokens=256,
        llm_timeout_seconds=1.0,
        llm_max_retries=1,
        workspace=tmp_path,
        log_level="DEBUG",
        host="127.0.0.1",
        port=8123,
    )
    return configured
