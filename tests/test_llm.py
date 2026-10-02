from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import anthropic
import httpx
import pytest

from furrow.config import Provider, Settings
from furrow.llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    IGNORED_DIRS,
    IGNORED_SUFFIXES,
    LLMClient,
    TruncatedResponseError,
    extract_json,
    is_transient_error,
)


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://example.invalid/v1/messages")


def _api_status_error(status: int) -> anthropic.APIStatusError:
    return anthropic.APIStatusError(
        "synthetic",
        response=httpx.Response(status, request=_request()),
        body=None,
    )


# --- extract_json ----------------------------------------------------------


def test_extract_json_plain() -> None:
    assert extract_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_extract_json_fenced_with_language_tag() -> None:
    text = 'Here you go:\n```json\n{"tasks": [{"id": "t1"}]}\n```\nHope that helps.'
    assert extract_json(text) == {"tasks": [{"id": "t1"}]}


def test_extract_json_fenced_uppercase_tag() -> None:
    assert extract_json('```JSON\n{"ok": true}\n```') == {"ok": True}


def test_extract_json_fenced_without_tag() -> None:
    assert extract_json('```\n{"ok": true}\n```') == {"ok": True}


def test_extract_json_wrapped_in_prose() -> None:
    text = (
        "Sure! Here is the plan you asked for.\n"
        'The plan is: {"tasks": [], "rationale": "because"}\n'
        "Let me know if you want changes."
    )
    assert extract_json(text) == {"tasks": [], "rationale": "because"}


def test_extract_json_nested_braces() -> None:
    payload: dict[str, Any] = {"outer": {"inner": {"deep": [1, 2, {"x": 1}]}}, "top": 1}
    text = f"noise before\n{json.dumps(payload)}\nnoise after"
    assert extract_json(text) == payload


def test_extract_json_braces_inside_string_literal() -> None:
    assert extract_json('{"a": "}{"}') == {"a": "}{"}


def test_extract_json_escaped_quotes_and_backslashes() -> None:
    assert extract_json(r'{"a": "say \"hi\"", "b": "c:\\tmp"}') == {
        "a": 'say "hi"',
        "b": "c:\\tmp",
    }


def test_extract_json_top_level_array() -> None:
    assert extract_json("result: [1, 2, {\"k\": [3]}] done") == [1, 2, {"k": [3]}]


def test_extract_json_skips_leading_invalid_brace() -> None:
    # The prose contains a stray "{" before the real payload.
    text = 'Use the format {id: string}. Concretely: {"id": "t1", "done": false}'
    assert extract_json(text) == {"id": "t1", "done": False}


def test_extract_json_failure_includes_offending_text() -> None:
    text = "there is definitely no JSON in this reply at all"
    with pytest.raises(ValueError) as excinfo:
        extract_json(text)
    assert text in str(excinfo.value)


def test_extract_json_failure_on_malformed_object() -> None:
    text = "{'single': 'quotes', nope}"
    with pytest.raises(ValueError) as excinfo:
        extract_json(text)
    assert "'single'" in str(excinfo.value)
    assert "parse error" in str(excinfo.value)


# --- errors / constants ----------------------------------------------------


def test_truncated_response_error_is_runtime_error() -> None:
    assert issubclass(TruncatedResponseError, RuntimeError)


def test_module_constants() -> None:
    assert DEFAULT_REQUEST_TIMEOUT_SECONDS == 120.0
    assert DEFAULT_MAX_TOKENS == 4096
    assert {".git", "node_modules", "__pycache__", "venv"} <= set(IGNORED_DIRS)
    assert IGNORED_SUFFIXES == (".pyc", ".pyo")


# --- list_files ------------------------------------------------------------


def test_list_files_prunes_ignored_trees(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hi')")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_text("junk")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "a.js").write_text("x")
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "lib.py").write_text("x")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x")
    (tmp_path / "src" / "stale.pyc").write_text("x")

    files = LLMClient(Settings()).list_files(tmp_path)

    assert files == ["main.py", "src/app.py"]
    for ignored in ("config", "a.js", "lib.py", "x.pyc", "stale.pyc", "node_modules", ".git"):
        assert not any(ignored in name for name in files)


def test_list_files_missing_directory_returns_empty(tmp_path: Path) -> None:
    assert LLMClient(Settings()).list_files(tmp_path / "nope") == []


# --- provider routing ------------------------------------------------------


async def test_complete_routes_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient(Settings(provider=Provider.OLLAMA, model="llama3"))
    calls: list[tuple[str, str, str]] = []

    async def fake(prompt: str, system: str, model: str) -> str:
        calls.append((prompt, system, model))
        return "ollama says hi"

    monkeypatch.setattr(client, "_complete_ollama", fake)
    assert await client.complete("hello", system="be nice") == "ollama says hi"
    assert calls == [("hello", "be nice", "llama3")]


@pytest.mark.parametrize(
    ("provider", "attr"),
    [(Provider.ANTHROPIC, "_complete_anthropic"), (Provider.OPENAI, "_complete_openai")],
)
async def test_complete_routes_to_cloud_providers(
    monkeypatch: pytest.MonkeyPatch, provider: Provider, attr: str
) -> None:
    client = LLMClient(Settings(provider=provider))
    seen: list[str] = []

    async def fake(prompt: str, system: str, model: str) -> str:
        seen.append(attr)
        return attr

    monkeypatch.setattr(client, attr, fake)
    assert await client.complete("hi") == attr
    assert seen == [attr]


async def test_complete_unknown_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient(Settings())
    monkeypatch.setattr(client.settings, "provider", "definitely-not-a-provider", raising=False)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hello")


# --- truncation detection --------------------------------------------------


async def test_anthropic_max_tokens_raises_truncated() -> None:
    client = LLMClient(Settings(provider=Provider.ANTHROPIC))

    class FakeMessages:
        async def create(self, **kwargs: object) -> object:
            block = type("Block", (), {"text": "half written"})()
            return type("Response", (), {"stop_reason": "max_tokens", "content": [block]})()

    client._anthropic = type("FakeAnthropic", (), {"messages": FakeMessages()})()

    with pytest.raises(TruncatedResponseError) as excinfo:
        await client.complete("write a long file", model="claude-test")
    assert "claude-test" in str(excinfo.value)
    assert "4096" in str(excinfo.value)


async def test_openai_length_finish_reason_raises_truncated() -> None:
    client = LLMClient(Settings(provider=Provider.OPENAI))

    class FakeCompletions:
        async def create(self, **kwargs: object) -> object:
            message = type("Message", (), {"content": "half written"})()
            choice = type("Choice", (), {"finish_reason": "length", "message": message})()
            return type("Response", (), {"choices": [choice]})()

    completions = FakeCompletions()
    chat = type("Chat", (), {"completions": completions})()
    client._openai = type("FakeOpenAI", (), {"chat": chat})()

    with pytest.raises(TruncatedResponseError) as excinfo:
        await client.complete("write a long file", model="gpt-test")
    assert "gpt-test" in str(excinfo.value)


# --- retry classification --------------------------------------------------


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_is_transient_error_for_retryable_statuses(status: int) -> None:
    assert is_transient_error(_api_status_error(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_is_transient_error_false_for_client_errors(status: int) -> None:
    assert is_transient_error(_api_status_error(status)) is False


def test_is_transient_error_for_transport_failures() -> None:
    assert is_transient_error(httpx.ConnectError("down", request=_request())) is True
    assert is_transient_error(httpx.ReadTimeout("slow", request=_request())) is True
    assert is_transient_error(ValueError("plain")) is False


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("furrow.llm.RETRY_WAIT_MIN", 0.0)
    monkeypatch.setattr("furrow.llm.RETRY_WAIT_MAX", 0.0)


async def test_call_with_retry_retries_transient_then_succeeds() -> None:
    client = LLMClient(Settings())
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _api_status_error(503)
        return "recovered"

    assert await client._call_with_retry(operation) == "recovered"
    assert attempts == 3


async def test_call_with_retry_does_not_retry_auth_errors() -> None:
    client = LLMClient(Settings())
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise _api_status_error(401)

    with pytest.raises(anthropic.APIStatusError):
        await client._call_with_retry(operation)
    assert attempts == 1


async def test_call_with_retry_gives_up_after_three_attempts() -> None:
    client = LLMClient(Settings())
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("down", request=_request())

    with pytest.raises(httpx.ConnectError):
        await client._call_with_retry(operation)
    assert attempts == 3


# --- timeout plumbing ------------------------------------------------------


def test_request_timeout_falls_back_to_default() -> None:
    assert LLMClient(Settings()).request_timeout == DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_request_timeout_honours_setting() -> None:
    # ``request_timeout`` is optional: the client must work with or without the
    # field being present on the settings object.
    configured = SimpleNamespace(request_timeout=12.5)
    assert LLMClient(cast(Settings, configured)).request_timeout == 12.5


def test_request_timeout_ignores_invalid_values() -> None:
    assert LLMClient(cast(Settings, SimpleNamespace(request_timeout=None))).request_timeout == (
        DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
