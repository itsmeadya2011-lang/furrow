"""Tests for :mod:`furrow.llm`.

Everything here is hermetic: no network, no API keys, no real SDK clients and no
sleeping.  Provider SDKs are replaced with tiny stand-in objects that expose only
the attributes ``furrow.llm`` actually touches.
"""

from __future__ import annotations

import json
import types

import anthropic
import httpx
import pytest

from furrow import llm
from furrow.config import Settings
from furrow.config import settings as module_level_settings
from furrow.llm import LLMClient, extract_json

# --------------------------------------------------------------------------------------
# extract_json
# --------------------------------------------------------------------------------------


def test_extract_json_bare_object() -> None:
    assert extract_json('{"passed": true, "summary": "ok"}') == {"passed": True, "summary": "ok"}


def test_extract_json_json_fenced() -> None:
    text = 'Here you go:\n```json\n{"tasks": [], "rationale": "none"}\n```\nDone.'
    assert extract_json(text) == {"tasks": [], "rationale": "none"}


def test_extract_json_plain_fenced() -> None:
    assert extract_json('```\n[1, 2, 3]\n```') == [1, 2, 3]


def test_extract_json_prose_wrapped() -> None:
    text = "Sure, here is the plan you asked for.\n{\"rationale\": \"because\"}\nLet me know."
    assert extract_json(text) == {"rationale": "because"}


def test_extract_json_prose_with_braces_when_fenced() -> None:
    """Fenced payloads win even when the surrounding prose contains braces."""
    text = "Do not use {braces} in shell commands.\n```json\n{\"a\": 1}\n```"
    assert extract_json(text) == {"a": 1}


def test_extract_json_braces_inside_strings() -> None:
    payload = {"summary": "use { and } literally", "nested": "a } b { c"}
    assert extract_json(f"Result: {json.dumps(payload)}") == payload


def test_extract_json_escaped_quotes_inside_strings() -> None:
    assert extract_json(r'{"summary": "say \"hi\" then stop"}') == {
        "summary": 'say "hi" then stop'
    }


def test_extract_json_nested_objects() -> None:
    payload = {"outer": {"inner": [{"x": 1}, {"y": [2, 3]}]}}
    assert extract_json(f"```json\n{json.dumps(payload)}\n```") == payload


def test_extract_json_array_of_scalars() -> None:
    assert extract_json("[1, 2, 3]") == [1, 2, 3]
    assert extract_json('["a", "b"]') == ["a", "b"]


@pytest.mark.xfail(
    reason="furrow/llm.py:60-87 yields the first '{' block before ever trying '['",
    strict=False,
)
def test_extract_json_bare_array_of_objects() -> None:
    """A bare (unfenced) JSON array of objects should parse as an array."""
    assert extract_json('[{"id": "1"}, {"id": "2"}]') == [{"id": "1"}, {"id": "2"}]


@pytest.mark.parametrize(
    "bad", ["", "   \n\t ", "not json at all", "no braces here", "{ unbalanced"]
)
def test_extract_json_raises_value_error_on_unparseable(bad: str) -> None:
    with pytest.raises(ValueError):
        extract_json(bad)


def test_extract_json_error_message_includes_snippet() -> None:
    with pytest.raises(ValueError) as excinfo:
        extract_json("totally unparseable response text")
    assert "totally unparseable response text" in str(excinfo.value)


# --------------------------------------------------------------------------------------
# LLMClient construction
# --------------------------------------------------------------------------------------


def test_client_settings_falls_back_to_module_settings() -> None:
    """Regression: ``LLMClient(settings=None).settings`` must never be ``None``.

    Agents construct the client without passing settings and then read
    ``self.client.settings.<model>``; a ``None`` settings object breaks every call.
    """
    client = LLMClient()
    assert client.settings is not None
    assert client.settings is module_level_settings

    explicit_none = LLMClient(settings=None)
    assert explicit_none.settings is not None
    assert explicit_none.settings is module_level_settings


def test_client_uses_injected_settings(tmp_path) -> None:
    custom = Settings(workspace=tmp_path, model="my-model")
    client = LLMClient(custom)
    assert client.settings is custom
    assert client.settings.model == "my-model"


# --------------------------------------------------------------------------------------
# Anthropic text-block narrowing
# --------------------------------------------------------------------------------------


def _text_block(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(type="text", text=text)


def _install_anthropic(client: LLMClient, create) -> None:
    """Point ``client.anthropic`` at a stub without touching the real SDK."""
    client._anthropic = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=create)
    )


async def test_anthropic_concatenates_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(content=[_text_block("hello "), _text_block("world")])

    client = LLMClient(Settings())
    _install_anthropic(client, create)
    result = await client._complete_anthropic("prompt", "", "some-model")

    assert result == "hello world"
    assert calls[0]["model"] == "some-model"
    assert calls[0]["max_tokens"] == 4096
    assert calls[0]["messages"] == [{"role": "user", "content": "prompt"}]
    assert calls[0]["system"]


async def test_anthropic_skips_non_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    async def create(**kwargs):
        thinking = types.SimpleNamespace(type="thinking", thinking="hmm")
        return types.SimpleNamespace(content=[thinking, _text_block("answer")])

    client = LLMClient(Settings())
    _install_anthropic(client, create)
    assert await client._complete_anthropic("p", "", "m") == "answer"


async def test_anthropic_without_text_block_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No text block must raise ``ValueError``, never ``AttributeError``."""

    async def create(**kwargs):
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="tool_use", id="t1", name="bash")]
        )

    client = LLMClient(Settings())
    _install_anthropic(client, create)
    with pytest.raises(ValueError, match="no text block"):
        await client._complete_anthropic("p", "", "m")


async def test_anthropic_empty_content_list_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def create(**kwargs):
        return types.SimpleNamespace(content=[])

    client = LLMClient(Settings())
    _install_anthropic(client, create)
    with pytest.raises(ValueError):
        await client._complete_anthropic("p", "", "m")


# --------------------------------------------------------------------------------------
# OpenAI response handling
# --------------------------------------------------------------------------------------


def _install_openai(client: LLMClient, create) -> None:
    client._openai = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )


async def test_openai_none_content_returns_empty_string() -> None:
    async def create(**kwargs):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=None),
                    finish_reason="stop",
                )
            ]
        )

    client = LLMClient(Settings())
    _install_openai(client, create)
    assert await client._complete_openai("prompt", "", "gpt-model") == ""


async def test_openai_returns_content() -> None:
    async def create(**kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hi"))]
        )

    client = LLMClient(Settings())
    _install_openai(client, create)
    assert await client._complete_openai("prompt", "be brief", "gpt-model") == "hi"


async def test_openai_no_choices_raises_value_error() -> None:
    async def create(**kwargs):
        return types.SimpleNamespace(choices=[])

    client = LLMClient(Settings())
    _install_openai(client, create)
    with pytest.raises(ValueError, match="no choices"):
        await client._complete_openai("p", "", "m")


# --------------------------------------------------------------------------------------
# Ollama over a stubbed httpx client
# --------------------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, data: dict) -> None:
        self._data = data
        self.raise_for_status_calls = 0

    def raise_for_status(self) -> None:
        self.raise_for_status_calls += 1

    def json(self) -> dict:
        return self._data


def _install_httpx(monkeypatch: pytest.MonkeyPatch, data: dict) -> list[dict]:
    """Replace ``furrow.llm.httpx`` with a stub module and return a call recorder."""
    calls: list[dict] = []

    class _StubAsyncClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        async def post(self, url, json=None):
            calls.append({"url": url, "payload": json, "timeout": self.kwargs.get("timeout")})
            return _StubResponse(data)

    monkeypatch.setattr(
        llm, "httpx", types.SimpleNamespace(Timeout=httpx.Timeout, AsyncClient=_StubAsyncClient)
    )
    return calls


@pytest.mark.parametrize("base_url", ["http://localhost:11434", "http://ollama.local:1234/"])
async def test_ollama_posts_to_chat_endpoint(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    calls = _install_httpx(monkeypatch, {"message": {"content": "hi"}})

    client = LLMClient(Settings(ollama_base_url=base_url))
    result = await client._complete_ollama("prompt", "sys", "llama3")

    assert result == "hi"
    assert calls[0]["url"] == f"{base_url.rstrip('/')}/api/chat"
    assert calls[0]["payload"]["model"] == "llama3"
    assert calls[0]["payload"]["stream"] is False
    assert calls[0]["payload"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prompt"},
    ]


async def test_ollama_default_system_prompt_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_httpx(monkeypatch, {"message": {"content": ""}})
    client = LLMClient(Settings())

    assert await client._complete_ollama("prompt", "", "llama3") == ""
    assert calls[0]["payload"]["messages"][0]["content"]
    assert calls[0]["timeout"] is not None


async def test_ollama_missing_message_key_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_httpx(monkeypatch, {"done": True})
    client = LLMClient(Settings())
    assert await client._complete_ollama("p", "", "m") == ""


# --------------------------------------------------------------------------------------
# complete() provider dispatch
# --------------------------------------------------------------------------------------


async def test_complete_dispatches_to_configured_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    client = LLMClient(Settings())

    async def anthropic_path(prompt, system, model):
        seen.append("anthropic")
        return "a"

    async def openai_path(prompt, system, model):
        seen.append("openai")
        return "o"

    async def ollama_path(prompt, system, model):
        seen.append("ollama")
        return "ll"

    async def check(provider: str, expected: str) -> None:
        seen.clear()
        monkeypatch.setattr(client, "_complete_anthropic", anthropic_path)
        monkeypatch.setattr(client, "_complete_openai", openai_path)
        monkeypatch.setattr(client, "_complete_ollama", ollama_path)
        client.settings = Settings(provider=provider)
        assert await client.complete("p") == expected
        # ``seen`` records which provider path ran, not its return value.
        assert seen == [provider]

    for provider, expected in (("anthropic", "a"), ("openai", "o"), ("ollama", "ll")):
        await check(provider, expected)


async def test_complete_rejects_unknown_provider() -> None:
    client = LLMClient(Settings())
    client.settings = Settings()
    object.__setattr__(client.settings, "provider", "bogus-provider")
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("p")


# --------------------------------------------------------------------------------------
# Retry behaviour
# --------------------------------------------------------------------------------------


def _rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.RateLimitError(
        "slow down",
        response=httpx.Response(429, request=request),
        body=None,
    )


@pytest.fixture
def instant_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace tenacity's async sleep with an instant no-op and record the waits."""
    waits: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        waits.append(seconds)

    for method in ("_complete_anthropic", "_complete_openai", "_complete_ollama"):
        retry_obj = getattr(LLMClient, method).retry
        monkeypatch.setattr(retry_obj, "sleep", fake_sleep)
    return waits


@pytest.mark.parametrize(
    "method",
    ["_complete_anthropic", "_complete_openai", "_complete_ollama"],
)
def test_retry_decorator_is_present(method: str) -> None:
    retry_obj = getattr(LLMClient, method).retry
    assert retry_obj is not None
    assert retry_obj.stop.max_attempt_number == 4
    assert retry_obj.reraise is True


async def test_anthropic_retries_rate_limit_and_reraises(
    instant_retry_sleep: list[float],
) -> None:
    attempts = 0

    async def create(**kwargs):
        nonlocal attempts
        attempts += 1
        raise _rate_limit_error()

    client = LLMClient(Settings())
    _install_anthropic(client, create)

    with pytest.raises(anthropic.RateLimitError):
        await client._complete_anthropic("p", "", "m")

    assert attempts == 4
    assert len(instant_retry_sleep) == 3


async def test_anthropic_retry_succeeds_before_limit(instant_retry_sleep: list[float]) -> None:
    attempts = 0

    async def create(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _rate_limit_error()
        return types.SimpleNamespace(content=[_text_block("recovered")])

    client = LLMClient(Settings())
    _install_anthropic(client, create)

    assert await client._complete_anthropic("p", "", "m") == "recovered"
    assert attempts == 3


async def test_non_retryable_error_is_not_retried(instant_retry_sleep: list[float]) -> None:
    attempts = 0

    async def create(**kwargs):
        nonlocal attempts
        attempts += 1
        raise ValueError("Anthropic returned a response with no text block")

    client = LLMClient(Settings())
    _install_anthropic(client, create)

    with pytest.raises(ValueError):
        await client._complete_anthropic("p", "", "m")
    assert attempts == 1


# --------------------------------------------------------------------------------------
# Filesystem helpers (async, exercised against tmp_path)
# --------------------------------------------------------------------------------------


async def test_write_then_read_and_list_files(tmp_path) -> None:
    client = LLMClient(Settings(workspace=tmp_path))
    await client.write_file(tmp_path / "nested" / "a.txt", "alpha")
    await client.write_file(tmp_path / "nested" / "b.txt", "beta")

    assert await client.read_file(tmp_path / "nested" / "a.txt") == "alpha"
    assert sorted(client.list_files(tmp_path)) == ["nested/a.txt", "nested/b.txt"]
    assert client.list_files(tmp_path / "missing") == []
