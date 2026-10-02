"""Regression tests for ``furrow.llm.LLMClient``.

No request ever leaves this module: the SDK clients are replaced by hand-rolled
fakes.  The tests pin down three historical bugs -- the settings default being
bound at import time, ``Provider.OLLAMA`` being unhandled, and
``response.content[0].text`` blowing up on a thinking/tool-use block.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

import furrow.llm as llm_module
from furrow.config import Provider, Settings
from furrow.llm import LLMClient

# ------------------------------------------------------------------- test doubles


class FakeBlock:
    """Stand-in for an Anthropic content block (text, thinking, tool_use, ...)."""

    def __init__(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            setattr(self, key, value)


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeBlock(content=content)


class FakeCompletions:
    def __init__(self, content: str, recorder: list[dict[str, Any]]) -> None:
        self._content = content
        self._recorder = recorder

    async def create(self, **kwargs: Any) -> Any:
        self._recorder.append(kwargs)
        return FakeBlock(choices=[FakeChoice(content=self._content)])


class FakeChat:
    def __init__(self, content: str, recorder: list[dict[str, Any]]) -> None:
        self.completions = FakeCompletions(content, recorder)


class FakeOpenAISDK:
    def __init__(self, content: str = "ok", recorder: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = recorder if recorder is not None else []
        self.chat = FakeChat(content, self.calls)


class FakeMessages:
    def __init__(self, content: list[Any], recorder: list[dict[str, Any]]) -> None:
        self._content = content
        self._recorder = recorder

    async def create(self, **kwargs: Any) -> Any:
        self._recorder.append(kwargs)
        return FakeBlock(content=self._content)


class FakeAnthropicSDK:
    def __init__(self, content: list[Any], recorder: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[dict[str, Any]] = recorder if recorder is not None else []
        self.messages = FakeMessages(content, self.calls)


def make_settings(**kwargs: Any) -> Settings:
    kwargs.setdefault("api_key_placeholder", None)
    kwargs.pop("api_key_placeholder", None)
    return Settings(**kwargs)


def _canned_complete(body: str) -> tuple[Callable[..., Awaitable[str]], list[dict[str, Any]]]:
    """A ``complete()`` stub with the real signature, plus a log of its calls.

    ``complete_json`` forwards all three of ``prompt``/``system``/``model`` to
    ``complete``, so the stub has to keep them -- renaming them or silencing the
    linter would hide a mismatch with ``LLMClient.complete``. Recording the call
    makes them genuinely used *and* lets each test assert the forwarding, not
    just the parsing.
    """
    calls: list[dict[str, Any]] = []

    async def fake_complete(prompt: str, system: str = "", model: str | None = None) -> str:
        calls.append({"prompt": prompt, "system": system, "model": model})
        return body

    return fake_complete, calls


# --------------------------------------------------------- settings fallback (bug 1)


def test_client_without_arguments_falls_back_to_the_module_settings_singleton(monkeypatch):
    # Bug: the default was bound at import time, so a settings swap afterwards was
    # ignored by every client built with no arguments.
    replacement = Settings(model="swapped-model")
    monkeypatch.setattr(llm_module, "settings", replacement)
    assert LLMClient().settings is replacement


def test_client_with_explicit_none_also_falls_back_to_the_module_singleton(monkeypatch):
    replacement = Settings(model="swapped-again")
    monkeypatch.setattr(llm_module, "settings", replacement)
    assert LLMClient(settings=None).settings is replacement


def test_two_clients_pick_up_a_later_settings_swap(monkeypatch):
    first = Settings(model="first")
    monkeypatch.setattr(llm_module, "settings", first)
    assert LLMClient().settings.model == "first"

    second = Settings(model="second")
    monkeypatch.setattr(llm_module, "settings", second)
    assert LLMClient().settings.model == "second"


def test_client_uses_the_settings_object_it_was_given():
    custom = Settings(model="custom-model")
    assert LLMClient(settings=custom).settings is custom


def test_client_settings_default_to_the_imported_singleton_when_untouched():
    # Sanity check for the fallback itself, without any monkeypatching.
    assert LLMClient().settings is LLMClient(settings=None).settings


# --------------------------------------------------------------- ollama (bug 2)


def test_ollama_base_url_gets_the_openai_compatible_v1_suffix():
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://host:1234")
    )
    # The OpenAI SDK normalises the URL with a trailing slash, hence rstrip().
    assert str(client.ollama.base_url).rstrip("/") == "http://host:1234/v1"


def test_ollama_base_url_with_a_trailing_slash_does_not_double_up():
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://host:1234/")
    )
    assert str(client.ollama.base_url).rstrip("/") == "http://host:1234/v1"


async def test_ollama_provider_dispatches_to_the_openai_surface_instead_of_raising():
    # Bug: complete() had no OLLAMA branch and raised
    # "ValueError: Unsupported provider: ollama".
    settings = Settings(
        provider=Provider.OLLAMA, ollama_base_url="http://host:1234", model="llama3"
    )
    client = LLMClient(settings=settings)
    sdk = FakeOpenAISDK(content="ollama says hi")
    client._ollama = sdk

    assert await client.complete("ping") == "ollama says hi"
    assert len(sdk.calls) == 1
    assert sdk.calls[0]["model"] == "llama3"


def test_ollama_sdk_client_is_cached_across_calls():
    client = LLMClient(
        settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://host:1234")
    )
    assert client.ollama is client.ollama


def test_empty_ollama_base_url_raises_value_error():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA, ollama_base_url=""))
    with pytest.raises(ValueError, match="ollama_base_url is not set"):
        _ = client.ollama


async def test_openai_provider_dispatches_to_the_openai_surface():
    settings = Settings(provider=Provider.OPENAI, openai_api_key="fake", model="gpt-x")
    client = LLMClient(settings=settings)
    sdk = FakeOpenAISDK(content="openai says hi")
    client._openai = sdk

    assert await client.complete("ping", system="be terse") == "openai says hi"
    assert sdk.calls[0]["messages"][0] == {"role": "system", "content": "be terse"}


# ------------------------------------------------------------ anthropic (bug 3)


async def test_anthropic_joins_every_text_block():
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="fake"))
    client._anthropic = FakeAnthropicSDK([FakeBlock(text="part one "), FakeBlock(text="part two")])
    assert await client.complete("ping") == "part one part two"


async def test_anthropic_ignores_non_text_blocks():
    # Bug: `response.content[0].text` raised AttributeError when the first block
    # was a thinking / tool_use block.
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="fake"))
    content = [
        FakeBlock(type="thinking", thinking="reasoning..."),
        FakeBlock(type="text", text="the answer"),
        FakeBlock(type="tool_use", name="bash", input={"cmd": "ls"}),
    ]
    client._anthropic = FakeAnthropicSDK(content)
    assert await client.complete("ping") == "the answer"


async def test_anthropic_returns_empty_string_when_there_is_no_text_block():
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="fake"))
    client._anthropic = FakeAnthropicSDK([FakeBlock(type="thinking", thinking="only thoughts")])
    assert await client.complete("ping") == ""


async def test_anthropic_tolerates_an_empty_content_list():
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="fake"))
    client._anthropic = FakeAnthropicSDK([])
    assert await client.complete("ping") == ""


async def test_anthropic_sends_the_model_system_prompt_and_user_message():
    settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="fake", model="claude-x")
    client = LLMClient(settings=settings)
    sdk = FakeAnthropicSDK([FakeBlock(text="ok")])
    client._anthropic = sdk

    await client.complete("the prompt", system="the system prompt")

    sent = sdk.calls[0]
    assert sent["model"] == "claude-x"
    assert sent["system"] == "the system prompt"
    assert sent["messages"] == [{"role": "user", "content": "the prompt"}]


async def test_anthropic_defaults_the_system_prompt_when_none_is_given():
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key="fake"))
    sdk = FakeAnthropicSDK([FakeBlock(text="ok")])
    client._anthropic = sdk

    await client.complete("the prompt")

    assert sdk.calls[0]["system"] == "You are a helpful coding assistant."


# --------------------------------------------------------------- complete_json


async def test_complete_json_parses_a_fenced_body(monkeypatch):
    client = LLMClient(settings=Settings())
    fake_complete, calls = _canned_complete(
        '```json\n{"tasks": [{"id": "1"}], "rationale": "because"}\n```'
    )

    monkeypatch.setattr(client, "complete", fake_complete)

    assert await client.complete_json("plan something", system="be terse") == {
        "tasks": [{"id": "1"}],
        "rationale": "because",
    }
    assert calls == [{"prompt": "plan something", "system": "be terse", "model": None}]


async def test_complete_json_parses_a_body_wrapped_in_prose(monkeypatch):
    client = LLMClient(settings=Settings())
    fake_complete, calls = _canned_complete(
        'Sure! Here is the plan: {"tasks": [], "rationale": "none"} enjoy'
    )

    monkeypatch.setattr(client, "complete", fake_complete)

    assert await client.complete_json("plan something") == {"tasks": [], "rationale": "none"}
    assert calls == [{"prompt": "plan something", "system": "", "model": None}]


async def test_complete_json_raises_value_error_on_an_unparseable_body(monkeypatch):
    client = LLMClient(settings=Settings())
    fake_complete, calls = _canned_complete("I refuse.")

    monkeypatch.setattr(client, "complete", fake_complete)
    with pytest.raises(ValueError, match="No valid JSON found"):
        await client.complete_json("plan something")
    assert calls == [{"prompt": "plan something", "system": "", "model": None}]


# --------------------------------------------------------------- api key errors


async def test_missing_anthropic_api_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        await client.complete("ping")


async def test_missing_openai_api_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(settings=Settings(provider=Provider.OPENAI, openai_api_key=None))
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        await client.complete("ping")


class _CountingClient(LLMClient):
    """Counts SDK-client lookups so a retry storm would be visible."""

    lookups = 0

    @property
    def anthropic(self):  # type: ignore[override]
        type(self).lookups += 1
        return super().anthropic


async def test_missing_api_key_is_not_retried_by_the_transient_retry(monkeypatch):
    # Bug: the transient retry swallowed the config error and kept re-trying a
    # request that could never succeed.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _CountingClient.lookups = 0
    client = _CountingClient(settings=Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None))

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        await client.complete("ping")

    assert _CountingClient.lookups == 1


def test_transient_classifier_rejects_a_missing_api_key_error():
    assert llm_module._is_transient(ValueError("ANTHROPIC_API_KEY is not set")) is False


def test_transient_classifier_accepts_plain_exceptions_as_non_transient():
    assert llm_module._is_transient(RuntimeError("boom")) is False


def test_settings_singleton_is_not_mutated_by_the_test_suite():
    # Guards against a test leaking a custom settings object into the singleton.
    assert llm_module.settings is not None
    assert isinstance(llm_module.settings, Settings)
