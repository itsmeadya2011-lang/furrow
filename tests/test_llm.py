from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import APIConnectionError as AnthropicConnectionError
from anthropic import APITimeoutError as AnthropicTimeoutError
from anthropic import AuthenticationError as AnthropicAuthError
from anthropic import BadRequestError as AnthropicBadRequest
from anthropic import InternalServerError as AnthropicServerError
from anthropic import PermissionDeniedError as AnthropicPermissionDenied
from anthropic import RateLimitError as AnthropicRateLimitError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APITimeoutError as OpenAITimeoutError
from openai import AuthenticationError as OpenAIAuthError
from openai import BadRequestError as OpenAIBadRequest
from openai import InternalServerError as OpenAIServerError
from openai import PermissionDeniedError as OpenAIPermissionDenied
from openai import RateLimitError as OpenAIRateLimitError

from furrow.config import Provider, Settings
from furrow.llm import (
    LLMAuthError,
    LLMClient,
    LLMProtocolError,
    LLMRequestError,
    LLMTransientError,
    _anthropic_text,
    _is_transient,
    _openai_text,
)


def make_settings(**overrides):
    return Settings(_env_file=None, **overrides)


def _anthropic_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.stop_reason = "end_turn"
    return resp


def _openai_response(text: str, finish_reason: str = "stop"):
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Settings defaults
# ---------------------------------------------------------------------------


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.planner_model == "claude-3-5-haiku-20241022"
    assert s.worker_model == "claude-3-5-sonnet-20241022"
    assert s.tester_model == "claude-3-5-sonnet-20241022"
    assert s.anthropic_api_key is None
    assert s.openai_api_key is None
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.max_tokens == 4096
    assert s.temperature == 0.2
    assert s.request_timeout == 120.0
    assert s.max_retries == 2
    assert s.max_retry_wait == 20.0
    assert s.test_timeout == 120


def test_settings_none_uses_singleton():
    client = LLMClient(settings=None)
    from furrow.config import settings as default_settings

    assert client.settings is default_settings


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


def test_complete_dispatches_anthropic():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.ANTHROPIC, anthropic_api_key="test"
        )
    )
    client._anthropic = MagicMock()
    client._anthropic.messages.create = AsyncMock(
        return_value=_anthropic_response("hi")
    )
    result = asyncio.run(client.complete("hello"))
    assert result == "hi"
    client._anthropic.messages.create.assert_called_once()
    kwargs = client._anthropic.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-20250514"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["temperature"] == 0.2


def test_complete_dispatches_openai():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.OPENAI, openai_api_key="test"
        )
    )
    client._openai = MagicMock()
    client._openai.chat.completions.create = AsyncMock(
        return_value=_openai_response("hi")
    )
    result = asyncio.run(client.complete("hello"))
    assert result == "hi"
    client._openai.chat.completions.create.assert_called_once()
    kwargs = client._openai.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-20250514"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["temperature"] == 0.2


def test_complete_dispatches_ollama():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "hi"}})

    client = LLMClient(
        settings=make_settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
        )
    )
    client._http_client = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    result = asyncio.run(client.complete("hello"))
    assert result == "hi"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["model"] == "claude-sonnet-4-20250514"
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["messages"][1]["role"] == "user"
    assert captured["body"]["messages"][1]["content"] == "hello"


# ---------------------------------------------------------------------------
# Ollama failure modes
# ---------------------------------------------------------------------------


def test_ollama_non_2xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    client = LLMClient(
        settings=make_settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
        )
    )
    client._http_client = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMRequestError, match="rejected the request"):
        asyncio.run(client.complete("hello"))


def test_ollama_non_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client = LLMClient(
        settings=make_settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
        )
    )
    client._http_client = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMProtocolError, match="non-JSON response"):
        asyncio.run(client.complete("hello"))


def test_ollama_missing_message():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True})

    client = LLMClient(
        settings=make_settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
        )
    )
    client._http_client = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMProtocolError, match="no 'message' object"):
        asyncio.run(client.complete("hello"))


def test_ollama_empty_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": ""}})

    client = LLMClient(
        settings=make_settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
        )
    )
    client._http_client = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMProtocolError, match="empty message content"):
        asyncio.run(client.complete("hello"))


def test_ollama_unreachable_host():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TransportError("connection refused")

    client = LLMClient(
        settings=make_settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
        )
    )
    client._http_client = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    with pytest.raises(LLMTransientError, match="Could not reach Ollama"):
        asyncio.run(client.complete("hello"))


# ---------------------------------------------------------------------------
# Anthropic multi-block extraction
# ---------------------------------------------------------------------------


def test_anthropic_multi_block():
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "hello "
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.text = None
    resp = MagicMock()
    resp.content = [text_block, tool_block, text_block]
    resp.stop_reason = "end_turn"
    assert _anthropic_text(resp, "claude-test") == "hello hello"


def test_anthropic_empty_content_raises():
    resp = MagicMock()
    resp.content = []
    resp.stop_reason = "end_turn"
    with pytest.raises(LLMProtocolError, match="no text content"):
        _anthropic_text(resp, "claude-test")


# ---------------------------------------------------------------------------
# OpenAI content=None handling
# ---------------------------------------------------------------------------


def test_openai_content_none():
    message = MagicMock()
    message.content = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "length"
    resp = MagicMock()
    resp.choices = [choice]
    with pytest.raises(LLMProtocolError, match="no text content"):
        _openai_text(resp, "gpt-test")


# ---------------------------------------------------------------------------
# Retry semantics
# ---------------------------------------------------------------------------


def _make_rate_limit(provider: str, status: int = 429):
    req = httpx.Request("POST", "https://api.example.com/v1/chat")
    resp = httpx.Response(status, request=req)
    if provider == "anthropic":
        return AnthropicRateLimitError(
            "rate limited", response=resp, body=None
        )
    return OpenAIRateLimitError("rate limited", response=resp, body=None)


def _make_auth_error(provider: str, status: int = 401):
    req = httpx.Request("POST", "https://api.example.com/v1/chat")
    resp = httpx.Response(status, request=req)
    if provider == "anthropic":
        return AnthropicAuthError(
            "unauthorized", response=resp, body=None
        )
    return OpenAIAuthError("unauthorized", response=resp, body=None)


def _make_server_error(provider: str, status: int = 500):
    req = httpx.Request("POST", "https://api.example.com/v1/chat")
    resp = httpx.Response(status, request=req)
    if provider == "anthropic":
        return AnthropicServerError(
            "server error", response=resp, body=None
        )
    return OpenAIServerError("server error", response=resp, body=None)


def _make_bad_request(provider: str):
    req = httpx.Request("POST", "https://api.example.com/v1/chat")
    resp = httpx.Response(400, request=req)
    if provider == "anthropic":
        return AnthropicBadRequest(
            "bad request", response=resp, body=None
        )
    return OpenAIBadRequest("bad request", response=resp, body=None)


def _make_permission_denied(provider: str):
    """A real 403 from either SDK raises PermissionDeniedError, not AuthenticationError."""
    req = httpx.Request("POST", "https://api.example.com/v1/chat")
    resp = httpx.Response(403, request=req)
    if provider == "anthropic":
        return AnthropicPermissionDenied(
            "forbidden", response=resp, body=None
        )
    return OpenAIPermissionDenied("forbidden", response=resp, body=None)


def _make_connection_error(provider: str):
    req = httpx.Request("POST", "https://api.example.com/v1/chat")
    if provider == "anthropic":
        return AnthropicConnectionError(request=req)
    return OpenAIConnectionError(request=req)


def _make_timeout_error(provider: str):
    req = httpx.Request("POST", "https://api.example.com/v1/chat")
    if provider == "anthropic":
        return AnthropicTimeoutError(request=req)
    return OpenAITimeoutError(request=req)


class StatusCarrier(Exception):
    """An error that only carries ``status_code``, with no recognised SDK type."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"unclassified HTTP {status_code}")
        self.status_code = status_code


def _anthropic_client(side_effect, **overrides):
    """Anthropic client whose transport raises ``side_effect`` (an error, or a list)."""
    defaults = {
        "provider": Provider.ANTHROPIC,
        "anthropic_api_key": "test",
        "max_retries": 2,
        "max_retry_wait": 0.0,
    }
    client = LLMClient(settings=make_settings(**(defaults | overrides)))
    client._anthropic = MagicMock()
    client._anthropic.messages.create = AsyncMock(side_effect=side_effect)
    return client


def _openai_client(side_effect, **overrides):
    """OpenAI client whose transport raises ``side_effect`` (an error, or a list)."""
    defaults = {
        "provider": Provider.OPENAI,
        "openai_api_key": "test",
        "max_retries": 2,
        "max_retry_wait": 0.0,
    }
    client = LLMClient(settings=make_settings(**(defaults | overrides)))
    client._openai = MagicMock()
    client._openai.chat.completions.create = AsyncMock(side_effect=side_effect)
    return client


def test_retry_429_anthropic():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test",
            max_retries=2,
            max_retry_wait=0.0,
        )
    )
    client._anthropic = MagicMock()
    err = _make_rate_limit("anthropic")
    client._anthropic.messages.create = AsyncMock(
        side_effect=[err, err, _anthropic_response("ok")]
    )
    result = asyncio.run(client.complete("hello"))
    assert result == "ok"
    assert client._anthropic.messages.create.call_count == 3


def test_retry_5xx_anthropic():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test",
            max_retries=2,
            max_retry_wait=0.0,
        )
    )
    client._anthropic = MagicMock()
    err = _make_server_error("anthropic", 503)
    client._anthropic.messages.create = AsyncMock(
        side_effect=[err, err, _anthropic_response("ok")]
    )
    result = asyncio.run(client.complete("hello"))
    assert result == "ok"
    assert client._anthropic.messages.create.call_count == 3


def test_retry_timeout_anthropic():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test",
            max_retries=2,
            max_retry_wait=0.0,
        )
    )
    client._anthropic = MagicMock()
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = AnthropicTimeoutError(request=req)
    client._anthropic.messages.create = AsyncMock(
        side_effect=[err, err, _anthropic_response("ok")]
    )
    result = asyncio.run(client.complete("hello"))
    assert result == "ok"
    assert client._anthropic.messages.create.call_count == 3


def test_no_retry_401_anthropic():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test",
            max_retries=2,
            max_retry_wait=0.0,
        )
    )
    client._anthropic = MagicMock()
    err = _make_auth_error("anthropic", 401)
    client._anthropic.messages.create = AsyncMock(side_effect=err)
    with pytest.raises(
        LLMAuthError,
        match=r"FURROW_ANTHROPIC_API_KEY.*ANTHROPIC_API_KEY",
    ):
        asyncio.run(client.complete("hello"))
    assert client._anthropic.messages.create.call_count == 1


def test_no_retry_403_anthropic():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test",
            max_retries=2,
            max_retry_wait=0.0,
        )
    )
    client._anthropic = MagicMock()
    err = _make_auth_error("anthropic", 403)
    client._anthropic.messages.create = AsyncMock(side_effect=err)
    with pytest.raises(
        LLMAuthError,
        match=r"FURROW_ANTHROPIC_API_KEY.*ANTHROPIC_API_KEY",
    ):
        asyncio.run(client.complete("hello"))
    assert client._anthropic.messages.create.call_count == 1


def test_no_retry_400_anthropic():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test",
            max_retries=2,
            max_retry_wait=0.0,
        )
    )
    client._anthropic = MagicMock()
    err = _make_bad_request("anthropic")
    client._anthropic.messages.create = AsyncMock(side_effect=err)
    with pytest.raises(LLMRequestError, match="rejected the request"):
        asyncio.run(client.complete("hello"))
    assert client._anthropic.messages.create.call_count == 1


def test_retry_openai_429():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.OPENAI,
            openai_api_key="test",
            max_retries=2,
            max_retry_wait=0.0,
        )
    )
    client._openai = MagicMock()
    err = _make_rate_limit("openai")
    client._openai.chat.completions.create = AsyncMock(
        side_effect=[err, err, _openai_response("ok")]
    )
    result = asyncio.run(client.complete("hello"))
    assert result == "ok"
    assert client._openai.chat.completions.create.call_count == 3


def test_no_retry_openai_401():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.OPENAI,
            openai_api_key="test",
            max_retries=2,
            max_retry_wait=0.0,
        )
    )
    client._openai = MagicMock()
    err = _make_auth_error("openai", 401)
    client._openai.chat.completions.create = AsyncMock(side_effect=err)
    with pytest.raises(
        LLMAuthError,
        match=r"FURROW_OPENAI_API_KEY.*OPENAI_API_KEY",
    ):
        asyncio.run(client.complete("hello"))
    assert client._openai.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# Transient / non-transient classification, per provider
#
# The providers differ: OpenAI had only 429-retried and 401-not-retried, so its
# 5xx, timeout, connection-reset and 403 paths were never exercised at all, and
# Anthropic never saw a connection reset or a genuine PermissionDeniedError.
# ---------------------------------------------------------------------------


def test_retry_connection_error_anthropic():
    err = _make_connection_error("anthropic")
    client = _anthropic_client([err, err, _anthropic_response("ok")])
    assert asyncio.run(client.complete("hello")) == "ok"
    assert client._anthropic.messages.create.call_count == 3


def test_no_retry_permission_denied_403_anthropic():
    """A real 403 arrives as PermissionDeniedError and must not be retried."""
    client = _anthropic_client(_make_permission_denied("anthropic"))
    with pytest.raises(LLMAuthError, match=r"rejected the credentials \(HTTP 403\)"):
        asyncio.run(client.complete("hello"))
    assert client._anthropic.messages.create.call_count == 1


def test_connection_error_becomes_transient_when_retries_are_exhausted():
    client = _anthropic_client(
        _make_connection_error("anthropic"), max_retries=0
    )
    with pytest.raises(LLMTransientError, match="FURROW_REQUEST_TIMEOUT"):
        asyncio.run(client.complete("hello"))
    assert client._anthropic.messages.create.call_count == 1


def test_retry_5xx_openai():
    err = _make_server_error("openai", 503)
    client = _openai_client([err, err, _openai_response("ok")])
    assert asyncio.run(client.complete("hello")) == "ok"
    assert client._openai.chat.completions.create.call_count == 3


def test_retry_timeout_openai():
    err = _make_timeout_error("openai")
    client = _openai_client([err, err, _openai_response("ok")])
    assert asyncio.run(client.complete("hello")) == "ok"
    assert client._openai.chat.completions.create.call_count == 3


def test_retry_connection_error_openai():
    err = _make_connection_error("openai")
    client = _openai_client([err, err, _openai_response("ok")])
    assert asyncio.run(client.complete("hello")) == "ok"
    assert client._openai.chat.completions.create.call_count == 3


def test_no_retry_permission_denied_403_openai():
    client = _openai_client(_make_permission_denied("openai"))
    with pytest.raises(LLMAuthError, match=r"rejected the credentials \(HTTP 403\)"):
        asyncio.run(client.complete("hello"))
    assert client._openai.chat.completions.create.call_count == 1


def test_no_retry_400_openai():
    client = _openai_client(_make_bad_request("openai"))
    with pytest.raises(LLMRequestError, match="rejected the request"):
        asyncio.run(client.complete("hello"))
    assert client._openai.chat.completions.create.call_count == 1


def test_connection_error_becomes_transient_when_retries_are_exhausted_openai():
    client = _openai_client(_make_connection_error("openai"), max_retries=0)
    with pytest.raises(LLMTransientError, match="FURROW_REQUEST_TIMEOUT"):
        asyncio.run(client.complete("hello"))
    assert client._openai.chat.completions.create.call_count == 1


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (LLMTransientError("boom"), True),
        (_make_rate_limit("anthropic"), True),
        (_make_rate_limit("openai"), True),
        (_make_server_error("anthropic", 500), True),
        (_make_server_error("openai", 503), True),
        (_make_connection_error("anthropic"), True),
        (_make_connection_error("openai"), True),
        (_make_timeout_error("anthropic"), True),
        (_make_timeout_error("openai"), True),
        (_make_auth_error("anthropic"), False),
        (_make_auth_error("openai"), False),
        (_make_permission_denied("anthropic"), False),
        (_make_permission_denied("openai"), False),
        (_make_bad_request("anthropic"), False),
        (_make_bad_request("openai"), False),
        (LLMAuthError("no key"), False),
        (LLMRequestError("bad request"), False),
        (LLMProtocolError("junk"), False),
        (ValueError("unrelated"), False),
    ],
)
def test_is_transient_classifies_every_provider_error(exc, transient):
    assert _is_transient(exc) is transient


@pytest.mark.parametrize(
    ("status", "transient"),
    [
        (408, True),
        (429, True),
        (500, True),
        (503, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
    ],
)
def test_is_transient_falls_back_to_the_status_code(status, transient):
    """Status-carrying errors of an unrecognised type still get classified."""
    assert _is_transient(StatusCarrier(status)) is transient


def test_max_retries_zero_still_makes_one_attempt():
    client = _anthropic_client(_make_rate_limit("anthropic"), max_retries=0)
    with pytest.raises(LLMTransientError, match="HTTP 429"):
        asyncio.run(client.complete("hello"))
    assert client._anthropic.messages.create.call_count == 1


def test_max_retries_counts_retries_not_attempts():
    client = _openai_client(_make_server_error("openai", 500), max_retries=3)
    with pytest.raises(LLMTransientError):
        asyncio.run(client.complete("hello"))
    assert client._openai.chat.completions.create.call_count == 4


# ---------------------------------------------------------------------------
# max_tokens / temperature forwarded
# ---------------------------------------------------------------------------


def test_anthropic_max_tokens_temperature_forwarded():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.ANTHROPIC,
            anthropic_api_key="test",
            max_tokens=1024,
            temperature=0.5,
        )
    )
    client._anthropic = MagicMock()
    client._anthropic.messages.create = AsyncMock(
        return_value=_anthropic_response("ok")
    )
    asyncio.run(client.complete("hello"))
    kwargs = client._anthropic.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 1024
    assert kwargs["temperature"] == 0.5


def test_openai_max_tokens_temperature_forwarded():
    client = LLMClient(
        settings=make_settings(
            provider=Provider.OPENAI,
            openai_api_key="test",
            max_tokens=512,
            temperature=0.7,
        )
    )
    client._openai = MagicMock()
    client._openai.chat.completions.create = AsyncMock(
        return_value=_openai_response("ok")
    )
    asyncio.run(client.complete("hello"))
    kwargs = client._openai.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 512
    assert kwargs["temperature"] == 0.7


def test_ollama_max_tokens_temperature_forwarded():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    client = LLMClient(
        settings=make_settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://localhost:11434",
        )
    )
    client._http_client = lambda: httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    asyncio.run(client.complete("hello"))
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["messages"][1]["role"] == "user"


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_list_files_prune(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".venv").mkdir()
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "foo.egg-info").mkdir()
    (tmp_path / "a.pyc").write_text("")
    (tmp_path / "b.py").write_text("")
    (tmp_path / "c.py").write_text("")
    result = LLMClient().list_files(tmp_path)
    assert result == ["b.py", "c.py"]


def test_list_files_sorted_posix(tmp_path: Path):
    (tmp_path / "z.py").write_text("")
    (tmp_path / "a.py").write_text("")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "m.py").write_text("")
    result = LLMClient().list_files(tmp_path)
    assert result == ["a.py", "sub/m.py", "z.py"]


def test_list_files_max_depth(tmp_path: Path):
    (tmp_path / "a.py").write_text("")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.py").write_text("")
    deep = sub / "deep"
    deep.mkdir()
    (deep / "c.py").write_text("")
    assert LLMClient().list_files(tmp_path) == [
        "a.py",
        "sub/b.py",
        "sub/deep/c.py",
    ]
    assert LLMClient().list_files(tmp_path, max_depth=0) == ["a.py"]
    assert LLMClient().list_files(tmp_path, max_depth=1) == ["a.py", "sub/b.py"]


def test_list_files_include(tmp_path: Path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    result = LLMClient().list_files(tmp_path, include=["*.py"])
    assert result == ["a.py"]


def test_list_files_missing_dir():
    assert LLMClient().list_files("/nonexistent/path/12345") == []


# ---------------------------------------------------------------------------
# read_file / write_file
# ---------------------------------------------------------------------------


def test_read_file_happy(tmp_path: Path):
    target = tmp_path / "file.txt"
    target.write_text("hello")
    result = asyncio.run(LLMClient().read_file(str(target)))
    assert result == "hello"


def test_read_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match=r"No such file: .*file\.txt"):
        asyncio.run(LLMClient().read_file(str(tmp_path / "file.txt")))


def test_read_file_is_directory(tmp_path: Path):
    (tmp_path / "dir").mkdir()
    with pytest.raises(
        IsADirectoryError,
        match=r"Expected a file but found a directory",
    ):
        asyncio.run(LLMClient().read_file(str(tmp_path / "dir")))


def test_write_file_happy(tmp_path: Path):
    target = tmp_path / "out.txt"
    asyncio.run(LLMClient().write_file(str(target), "hello"))
    assert target.read_text() == "hello"


def test_write_file_creates_parents(tmp_path: Path):
    target = tmp_path / "deep" / "dir" / "file.txt"
    asyncio.run(LLMClient().write_file(str(target), "hello"))
    assert target.read_text() == "hello"


# ---------------------------------------------------------------------------
# API key error messages
# ---------------------------------------------------------------------------


def test_anthropic_api_key_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(
        settings=make_settings(
            provider=Provider.ANTHROPIC, anthropic_api_key=None
        )
    )
    with pytest.raises(
        LLMAuthError,
        match=r"FURROW_ANTHROPIC_API_KEY.*ANTHROPIC_API_KEY",
    ):
        asyncio.run(client.complete("hello"))


def test_openai_api_key_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_OPENAI_API_KEY", raising=False)
    client = LLMClient(
        settings=make_settings(
            provider=Provider.OPENAI, openai_api_key=None
        )
    )
    with pytest.raises(
        LLMAuthError,
        match=r"FURROW_OPENAI_API_KEY.*OPENAI_API_KEY",
    ):
        asyncio.run(client.complete("hello"))
