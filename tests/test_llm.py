"""Regression tests for ``furrow.llm``.

Self-contained on purpose: no shared ``conftest.py``, no real provider clients,
and zero network. The Ollama path is exercised by injecting an
``httpx.AsyncClient`` backed by ``httpx.MockTransport`` into the client's
private cache, so the exact request (URL + JSON body) and the response handling
are asserted without a socket.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from furrow.config import Provider
from furrow.llm import DEFAULT_SYSTEM_PROMPT, LLMClient

Handler = Callable[[httpx.Request], httpx.Response]


def _settings(**overrides: Any) -> SimpleNamespace:
    """Duck-typed stand-in for ``furrow.config.Settings``.

    Only the attributes ``LLMClient`` actually reads are provided, so these
    tests stay valid when unrelated settings are added to the real model.
    """
    base: dict[str, Any] = {
        "provider": Provider.OLLAMA,
        "model": "stub-default-model",
        "ollama_model": "stub-ollama-model",
        "ollama_base_url": "http://ollama.test:11434",
        "anthropic_api_key": None,
        "openai_api_key": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _ok_payload(content: str = "ollama says hi") -> dict[str, Any]:
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "model": "stub-ollama-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _body_text(payload: Any) -> str:
    """The exact body text a mock handler serves, so errors can be asserted on."""
    return json.dumps(payload, separators=(",", ":"))


def _json_handler(
    payload: Any,
    status_code: int = 200,
    seen: list[httpx.Request] | None = None,
) -> Handler:
    def _handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(
            status_code,
            content=_body_text(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
        )

    return _handler


def _text_handler(
    body: str,
    status_code: int = 200,
    content_type: str = "text/plain",
) -> Handler:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=body, headers={"content-type": content_type})

    return _handler


def _sent_json(requests: list[httpx.Request], index: int = 0) -> dict[str, Any]:
    return json.loads(requests[index].content)


@pytest.fixture
async def build_client() -> AsyncIterator[Callable[..., LLMClient]]:
    """Factory for clients wired to a mock transport; always closed on teardown."""
    clients: list[LLMClient] = []

    def _build(handler: Handler | None = None, **overrides: Any) -> LLMClient:
        client = LLMClient(_settings(**overrides))
        if handler is not None:
            client._httpx = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    try:
        yield _build
    finally:
        for client in clients:
            await client.aclose()


# --------------------------------------------------------------------------- #
# _complete_ollama: request shape
# --------------------------------------------------------------------------- #

_BASE_URLS = [
    pytest.param("http://ollama.test:11434", "http://ollama.test:11434", id="no-slash"),
    pytest.param("http://ollama.test:11434/", "http://ollama.test:11434", id="one-slash"),
    pytest.param("http://ollama.test:11434///", "http://ollama.test:11434", id="many-slashes"),
]


@pytest.mark.parametrize(("configured", "expected_host"), _BASE_URLS)
async def test_complete_ollama_targets_openai_compatible_endpoint(
    build_client: Callable[..., LLMClient],
    configured: str,
    expected_host: str,
) -> None:
    """A trailing slash on the base URL never leaks into the request URL."""
    seen: list[httpx.Request] = []
    client = build_client(
        _json_handler(_ok_payload(), seen=seen), ollama_base_url=configured
    )

    await client._complete_ollama("say hi", "be terse", "llama3.1")

    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert str(seen[0].url) == f"{expected_host}/v1/chat/completions"


async def test_complete_ollama_sends_expected_payload_and_returns_content(
    build_client: Callable[..., LLMClient],
) -> None:
    seen: list[httpx.Request] = []
    client = build_client(
        _json_handler(_ok_payload("the answer is 42"), seen=seen),
        ollama_base_url="http://ollama.test:11434",
    )

    result = await client._complete_ollama("what is 6*7?", "answer briefly", "llama3.1:8b")

    assert result == "the answer is 42"
    body = _sent_json(seen)
    assert set(body) == {"model", "messages", "stream"}
    assert body["model"] == "llama3.1:8b"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "answer briefly"},
        {"role": "user", "content": "what is 6*7?"},
    ]


@pytest.mark.parametrize("system", ["", None])
async def test_complete_ollama_falls_back_to_default_system_prompt(
    build_client: Callable[..., LLMClient],
    system: str | None,
) -> None:
    seen: list[httpx.Request] = []
    client = build_client(_json_handler(_ok_payload(), seen=seen))

    await client._complete_ollama("hello", system, "llama3.1")  # type: ignore[arg-type]

    messages = _sent_json(seen)["messages"]
    assert messages[0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert messages[1] == {"role": "user", "content": "hello"}


async def test_complete_ollama_wraps_transport_failures(
    build_client: Callable[..., LLMClient],
) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = build_client(_handler, ollama_base_url="http://ollama.test:11434")

    with pytest.raises(RuntimeError) as excinfo:
        await client._complete_ollama("hi", "", "llama3.1")

    message = str(excinfo.value)
    assert "Ollama request to" in message
    assert "http://ollama.test:11434/v1/chat/completions" in message
    assert "connection refused" in message
    assert isinstance(excinfo.value.__cause__, httpx.HTTPError)


# --------------------------------------------------------------------------- #
# _complete_ollama: error handling
# --------------------------------------------------------------------------- #

_ERROR_STATUSES = [
    pytest.param(400, "model not found", id="400"),
    pytest.param(404, "no such path", id="404"),
    pytest.param(500, "internal error", id="500"),
    pytest.param(503, "model is loading", id="503"),
]


@pytest.mark.parametrize(("status_code", "body"), _ERROR_STATUSES)
async def test_complete_ollama_raises_with_status_code_on_error_response(
    build_client: Callable[..., LLMClient],
    status_code: int,
    body: str,
) -> None:
    client = build_client(_text_handler(body, status_code=status_code))

    with pytest.raises(RuntimeError) as excinfo:
        await client._complete_ollama("hi", "", "llama3.1")

    message = str(excinfo.value)
    assert f"HTTP {status_code}" in message
    assert body in message
    assert "http://ollama.test:11434/v1/chat/completions" in message


async def test_complete_ollama_raises_when_body_is_not_json(
    build_client: Callable[..., LLMClient],
) -> None:
    client = build_client(_text_handler("<html>gateway timeout</html>"))

    with pytest.raises(RuntimeError) as excinfo:
        await client._complete_ollama("hi", "", "llama3.1")

    message = str(excinfo.value)
    assert "Could not parse Ollama response" in message
    assert "gateway timeout" in message


_BAD_PAYLOADS = [
    pytest.param({}, id="empty-object"),
    pytest.param({"choices": []}, id="empty-choices"),
    pytest.param({"choices": [{}]}, id="missing-message"),
    pytest.param({"choices": [{"message": {}}]}, id="missing-content"),
    pytest.param({"choices": "not-a-list"}, id="choices-not-a-list"),
    pytest.param({"choices": [{"message": "not-a-dict"}]}, id="message-not-a-dict"),
    pytest.param({"model": "llama3.1"}, id="only-model"),
]


@pytest.mark.parametrize("payload", _BAD_PAYLOADS)
async def test_complete_ollama_raises_when_choices_are_missing_or_malformed(
    build_client: Callable[..., LLMClient],
    payload: Any,
) -> None:
    client = build_client(_json_handler(payload))

    with pytest.raises(RuntimeError) as excinfo:
        await client._complete_ollama("hi", "", "llama3.1")

    message = str(excinfo.value)
    assert "Could not parse Ollama response" in message
    assert _body_text(payload) in message


_NON_TEXT_CONTENTS = [
    pytest.param(None, id="null"),
    pytest.param(42, id="number"),
    pytest.param([{"text": "structured"}], id="content-blocks"),
    pytest.param({"text": "structured"}, id="object"),
]


@pytest.mark.parametrize("content", _NON_TEXT_CONTENTS)
async def test_complete_ollama_raises_when_content_is_not_text(
    build_client: Callable[..., LLMClient],
    content: Any,
) -> None:
    client = build_client(_json_handler({"choices": [{"message": {"content": content}}]}))

    with pytest.raises(RuntimeError) as excinfo:
        await client._complete_ollama("hi", "", "llama3.1")

    message = str(excinfo.value)
    assert "non-text content" in message
    assert repr(content) in message
    # The dedicated error must not masquerade as a JSON parse failure.
    assert "Could not parse" not in message


# --------------------------------------------------------------------------- #
# complete(): dispatch and model resolution
# --------------------------------------------------------------------------- #

_PRIVATE_COMPLETERS = ("_complete_ollama", "_complete_anthropic", "_complete_openai")

_PROVIDER_ROUTES = [
    pytest.param(Provider.OLLAMA, "_complete_ollama", id="ollama"),
    pytest.param(Provider.ANTHROPIC, "_complete_anthropic", id="anthropic"),
    pytest.param(Provider.OPENAI, "_complete_openai", id="openai"),
]


@pytest.mark.parametrize(("provider", "expected"), _PROVIDER_ROUTES)
async def test_complete_dispatches_to_the_provider_helper(
    build_client: Callable[..., LLMClient],
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    expected: str,
) -> None:
    client = build_client(_json_handler(_ok_payload()), provider=provider)
    calls: list[tuple[str, str, str, str]] = []

    def _recorder(name: str) -> Callable[..., Any]:
        async def _impl(prompt: str, system: str, model: str) -> str:
            calls.append((name, prompt, system, model))
            return f"from-{name}"

        return _impl

    for name in _PRIVATE_COMPLETERS:
        monkeypatch.setattr(client, name, _recorder(name))

    result = await client.complete("the prompt", "the system")

    assert result == f"from-{expected}"
    assert calls == [(expected, "the prompt", "the system", client.default_model())]


async def test_complete_rejects_an_unknown_provider(
    build_client: Callable[..., LLMClient],
) -> None:
    client = build_client(_json_handler(_ok_payload()), provider="gemini")

    with pytest.raises(ValueError) as excinfo:
        await client.complete("the prompt")

    assert "Unsupported provider" in str(excinfo.value)
    assert "gemini" in str(excinfo.value)
    # Nothing was sent anywhere for a provider we cannot serve.
    assert client._httpx is not None


def test_default_model_prefers_ollama_model_for_ollama() -> None:
    client = LLMClient(
        _settings(provider=Provider.OLLAMA, model="generic-model", ollama_model="llama3.1:8b")
    )
    assert client.default_model() == "llama3.1:8b"


@pytest.mark.parametrize("provider", [Provider.ANTHROPIC, Provider.OPENAI])
def test_default_model_uses_generic_model_for_other_providers(provider: Provider) -> None:
    client = LLMClient(
        _settings(provider=provider, model="generic-model", ollama_model="llama3.1:8b")
    )
    assert client.default_model() == "generic-model"


async def test_explicit_model_overrides_default_for_ollama(
    build_client: Callable[..., LLMClient],
) -> None:
    seen: list[httpx.Request] = []
    client = build_client(
        _json_handler(_ok_payload("ok"), seen=seen),
        provider=Provider.OLLAMA,
        model="generic-model",
        ollama_model="llama3.1:8b",
    )

    result = await client.complete("prompt", "system", model="explicit-override")

    assert result == "ok"
    assert client.default_model() == "llama3.1:8b"
    assert _sent_json(seen)["model"] == "explicit-override"


async def test_explicit_model_overrides_default_for_anthropic(
    build_client: Callable[..., LLMClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = build_client(provider=Provider.ANTHROPIC, model="generic-model")
    recorded: dict[str, str] = {}

    async def _fake_anthropic(prompt: str, system: str, model: str) -> str:
        recorded.update(prompt=prompt, system=system, model=model)
        return "ok"

    monkeypatch.setattr(client, "_complete_anthropic", _fake_anthropic)

    result = await client.complete("prompt", "system", model="explicit-override")

    assert result == "ok"
    assert client.default_model() == "generic-model"
    assert recorded == {
        "prompt": "prompt",
        "system": "system",
        "model": "explicit-override",
    }


# --------------------------------------------------------------------------- #
# Laziness and lifecycle
# --------------------------------------------------------------------------- #


def test_construction_creates_no_clients_and_needs_no_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = LLMClient(_settings())

    assert client._anthropic is None
    assert client._openai is None
    assert client._httpx is None
    # Reading configuration must not lazily spin up a transport either.
    assert client.default_model() == "stub-ollama-model"
    assert client._httpx is None


def test_provider_clients_report_a_missing_key_without_being_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(_settings())

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        _ = client.openai

    assert client._anthropic is None
    assert client._openai is None


async def test_aclose_closes_and_resets_the_http_client(
    build_client: Callable[..., LLMClient],
) -> None:
    client = build_client(_json_handler(_ok_payload()))
    transport_client = client._httpx
    assert transport_client is not None
    assert transport_client.is_closed is False

    await client.aclose()

    assert transport_client.is_closed is True
    assert client._httpx is None

    # Idempotent: a second close is a no-op, not an error.
    await client.aclose()
    assert client._httpx is None


async def test_aclose_is_a_noop_before_any_client_was_created() -> None:
    client = LLMClient(_settings())

    await client.aclose()
    await client.aclose()

    assert client._httpx is None


async def test_http_client_is_recreated_after_aclose() -> None:
    client = LLMClient(_settings())
    first = client.http
    assert client._httpx is first

    await client.aclose()

    second = client.http
    assert second is not first
    assert first.is_closed is True
    assert second.is_closed is False
    await client.aclose()


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #

_UNICODE_SAMPLE = "héllo — wörld ✓ 日本語\nsecond line\n\ttab"


async def test_write_file_creates_missing_parent_directories(tmp_path: Path) -> None:
    client = LLMClient(_settings())
    target = tmp_path / "a" / "b" / "c" / "note.txt"

    await client.write_file(target, "deep")

    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "deep"


async def test_file_helpers_round_trip_unicode(tmp_path: Path) -> None:
    client = LLMClient(_settings())
    target = tmp_path / "unicode.txt"

    await client.write_file(target, _UNICODE_SAMPLE)

    assert await client.read_file(target) == _UNICODE_SAMPLE
    assert target.read_bytes() == _UNICODE_SAMPLE.encode("utf-8")


async def test_write_file_overwrites_and_allows_empty_content(tmp_path: Path) -> None:
    client = LLMClient(_settings())
    target = tmp_path / "log.txt"

    await client.write_file(target, "first version")
    await client.write_file(target, "")

    assert await client.read_file(target) == ""
    assert target.stat().st_size == 0
    assert client.list_files(tmp_path) == ["log.txt"]


async def test_list_files_returns_posix_paths_relative_to_the_directory(
    tmp_path: Path,
) -> None:
    client = LLMClient(_settings())
    await client.write_file(tmp_path / "top.txt", "1")
    await client.write_file(tmp_path / "pkg" / "__init__.py", "2")
    await client.write_file(tmp_path / "pkg" / "sub" / "deep.py", "3")
    (tmp_path / "pkg" / "empty_dir").mkdir(parents=True, exist_ok=True)

    found = client.list_files(tmp_path)

    assert sorted(found) == ["pkg/__init__.py", "pkg/sub/deep.py", "top.txt"]
    # POSIX separators, and never absolute or parent-relative paths.
    assert all("\\" not in entry for entry in found)
    assert all(not Path(entry).is_absolute() for entry in found)
    assert "empty_dir" not in found


async def test_list_files_on_missing_directory_returns_empty_list(tmp_path: Path) -> None:
    client = LLMClient(_settings())
    assert client.list_files(tmp_path / "does" / "not" / "exist") == []


async def test_list_files_on_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    client = LLMClient(_settings())
    assert client.list_files(tmp_path) == []


async def test_file_helpers_accept_str_and_path_arguments(tmp_path: Path) -> None:
    client = LLMClient(_settings())
    as_path = tmp_path / "mixed" / "as_path.txt"
    as_str = tmp_path / "mixed" / "as_str.txt"

    await client.write_file(as_path, "written with a Path")
    await client.write_file(str(as_str), "written with a str")

    assert await client.read_file(as_path) == "written with a Path"
    assert await client.read_file(str(as_str)) == "written with a str"
    # A str directory argument yields the same relative POSIX listing.
    assert sorted(client.list_files(str(tmp_path))) == ["mixed/as_path.txt", "mixed/as_str.txt"]


async def test_read_file_on_missing_path_raises(tmp_path: Path) -> None:
    client = LLMClient(_settings())
    with pytest.raises(FileNotFoundError):
        await client.read_file(tmp_path / "nope.txt")
