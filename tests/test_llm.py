"""Regression tests for :mod:`furrow.llm` - the async client wrapper.

Everything here is hermetic:

* no provider SDK is ever constructed for real and no socket is opened - the
  Anthropic / OpenAI / Ollama paths are driven by in-process fakes attached with
  ``monkeypatch.setattr(client, "_anthropic", fake)`` (and, for Ollama, by
  replacing ``httpx.AsyncClient``);
* :class:`~furrow.config.Settings` instances come from the ``make_settings``
  fixture in ``tests/conftest.py``, i.e. built with ``_env_file=None`` after every
  ambient ``FURROW_*`` variable has been removed, so neither the developer's
  ``.env`` nor the shell can change an outcome;
* all filesystem work uses ``tmp_path`` / ``tmp_path_factory``.

The behaviours encoded below are the ones the implementation must guarantee.
Several guard bugs that used to be silent: truncated answers returned as if
complete, non-text Anthropic blocks crashing with ``AttributeError``, an empty
OpenAI reply collapsing to ``""`` and then resurfacing as a baffling JSON parse
error downstream, ``LLMClient(settings=None)`` storing ``None`` and making every
default-constructed agent raise ``AttributeError: 'NoneType' object has no
attribute 'planner_model'``, and ``list_files`` blowing up on a symlink that
escapes the scanned tree.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock
from anthropic.types import Usage as AnthropicUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as OpenAIChoice

from furrow.config import Provider
from furrow.config import settings as global_settings
from furrow.llm import DEFAULT_SYSTEM_PROMPT, LLMClient, LLMError, LLMResponseError

# ============================================================= provider doubles


@dataclass
class _ThinkingBlock:
    """Duck-typed stand-in for ``anthropic.types.ThinkingBlock`` - no ``.text``."""

    thinking: str = "let me think"


@dataclass
class _ToolUseBlock:
    """Duck-typed stand-in for ``anthropic.types.ToolUseBlock`` - no ``.text``."""

    name: str = "bash"


@dataclass
class _TextBlock:
    """Duck-typed stand-in for ``anthropic.types.TextBlock``."""

    text: str = ""


@dataclass
class _FakeAnthropicResponse:
    content: list[Any] = field(default_factory=list)
    stop_reason: str | None = "end_turn"


class _FakeAnthropicMessages:
    def __init__(self, response: _FakeAnthropicResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeAnthropicResponse:
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropic:
    """Stands in for ``AsyncAnthropic``: ``.messages.create`` plus ``close()``."""

    def __init__(self, response: _FakeAnthropicResponse) -> None:
        self.messages = _FakeAnthropicMessages(response)
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@dataclass
class _FakeOpenAIMessage:
    content: str | None = None
    refusal: str | None = None
    tool_calls: list[Any] | None = None


@dataclass
class _FakeOpenAIChoice:
    message: _FakeOpenAIMessage = field(default_factory=_FakeOpenAIMessage)
    finish_reason: str | None = "stop"


@dataclass
class _FakeOpenAIResponse:
    choices: Sequence[OpenAIChoice | _FakeOpenAIChoice] = field(default_factory=list)


class _FakeOpenAICompletions:
    def __init__(self, response: _FakeOpenAIResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeOpenAIResponse:
        self.calls.append(kwargs)
        return self._response


class _FakeOpenAIChat:
    def __init__(self, completions: _FakeOpenAICompletions) -> None:
        self.completions = completions


class _FakeOpenAI:
    """Stands in for ``AsyncOpenAI``: ``.chat.completions.create`` plus ``close()``."""

    def __init__(self, response: _FakeOpenAIResponse) -> None:
        self.completions = _FakeOpenAICompletions(response)
        self.chat = _FakeOpenAIChat(self.completions)
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _FakeHttpxResponse:
    """Stands in for ``httpx.Response`` - only what ``_complete_ollama`` touches."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        json_data: Any = None,
        json_error: BaseException | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json_data = json_data
        self._json_error = json_error

    def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._json_data


class _FakeHttpxClient:
    """Stands in for ``httpx.AsyncClient`` used as an async context manager."""

    def __init__(
        self,
        response: _FakeHttpxResponse,
        init_kwargs: dict[str, Any],
        post_error: BaseException | None = None,
    ) -> None:
        self._response = response
        self.init_kwargs = init_kwargs
        self.post_error = post_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> _FakeHttpxClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _FakeHttpxResponse:
        self.calls.append((url, kwargs))
        if self.post_error is not None:
            raise self.post_error
        return self._response


class _FakeHttpxFactory:
    """Callable replacement for ``httpx.AsyncClient`` that records every build."""

    def __init__(
        self, response: _FakeHttpxResponse, post_error: BaseException | None = None
    ) -> None:
        self._response = response
        self._post_error = post_error
        self.instances: list[_FakeHttpxClient] = []

    def __call__(self, **kwargs: Any) -> _FakeHttpxClient:
        client = _FakeHttpxClient(self._response, kwargs, self._post_error)
        self.instances.append(client)
        return client


# ===================================================================== helpers


def _attach_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    client: LLMClient,
    response: _FakeAnthropicResponse,
) -> _FakeAnthropic:
    fake = _FakeAnthropic(response)
    monkeypatch.setattr(client, "_anthropic", fake)
    return fake


def _attach_openai(
    monkeypatch: pytest.MonkeyPatch,
    client: LLMClient,
    response: _FakeOpenAIResponse,
) -> _FakeOpenAI:
    fake = _FakeOpenAI(response)
    monkeypatch.setattr(client, "_openai", fake)
    return fake


def _attach_ollama(
    monkeypatch: pytest.MonkeyPatch,
    response: _FakeHttpxResponse,
    post_error: BaseException | None = None,
) -> _FakeHttpxFactory:
    factory = _FakeHttpxFactory(response, post_error)
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return factory


def _ok_anthropic(text: str = "ok") -> _FakeAnthropicResponse:
    return _FakeAnthropicResponse(content=[_TextBlock(text=text)])


def _ok_openai(text: str = "ok") -> _FakeOpenAIResponse:
    return _FakeOpenAIResponse(choices=[_FakeOpenAIChoice(message=_FakeOpenAIMessage(text))])


def _real_anthropic_message(
    content: list[Any], stop_reason: str = "end_turn"
) -> AnthropicMessage:
    """Build a genuine ``anthropic.types.Message`` so the local fakes stay anchored."""
    return AnthropicMessage(
        id="msg_test",
        content=content,
        model="claude-test",
        role="assistant",
        stop_reason=stop_reason,  # type: ignore[arg-type]
        stop_sequence=None,
        type="message",
        usage=AnthropicUsage(input_tokens=1, output_tokens=2),
    )


def _real_openai_completion(
    message: ChatCompletionMessage, finish_reason: str = "stop"
) -> ChatCompletion:
    """Build a genuine ``openai.types.chat.ChatCompletion``."""
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[OpenAIChoice(finish_reason=finish_reason, index=0, message=message)],  # type: ignore[arg-type]
        created=0,
        model="gpt-test",
        object="chat.completion",
    )


# ========================================================== settings resolution


def test_llm_client_default_constructor_uses_global_settings_singleton() -> None:
    assert LLMClient().settings is global_settings


def test_llm_client_with_settings_none_falls_back_to_global_singleton() -> None:
    """The regression that made every default-constructed agent crash.

    ``furrow/agents/*`` forward ``settings=settings`` where their own parameter
    defaults to ``None``; storing that ``None`` verbatim made the very next line
    raise ``AttributeError: 'NoneType' object has no attribute 'planner_model'``.
    """
    client = LLMClient(settings=None)

    assert client.settings is global_settings
    assert client.settings is not None


def test_llm_client_without_settings_exposes_role_model_fields() -> None:
    client = LLMClient()

    assert isinstance(client.settings.planner_model, str)
    assert client.settings.planner_model
    assert isinstance(client.settings.worker_model, str)
    assert client.settings.worker_model
    assert isinstance(client.settings.tester_model, str)
    assert client.settings.tester_model


def test_llm_client_honours_explicit_settings_instance(
    make_settings: Callable[..., Any],
) -> None:
    explicit = make_settings(model="explicit-model")

    client = LLMClient(settings=explicit)

    assert client.settings is explicit
    assert client.settings.model == "explicit-model"
    assert client.settings is not global_settings


# =========================================================== exception hierarchy


def test_llm_response_error_mro_order_is_canonical() -> None:
    assert [cls.__name__ for cls in LLMResponseError.__mro__] == [
        "LLMResponseError",
        "LLMError",
        "RuntimeError",
        "ValueError",
        "Exception",
        "BaseException",
        "object",
    ]


def test_llm_response_error_is_both_llm_error_and_value_error() -> None:
    error = LLMResponseError("bad reply")

    assert isinstance(error, LLMError)
    assert isinstance(error, RuntimeError)
    assert isinstance(error, ValueError)


def test_llm_error_groups_under_runtime_error() -> None:
    assert issubclass(LLMError, RuntimeError)
    assert not issubclass(LLMError, ValueError)


def test_llm_response_error_is_caught_by_bare_value_error_handler() -> None:
    """``furrow/agents/tester.py`` still guards parsing with ``except ValueError``."""

    with pytest.raises(ValueError):
        raise LLMResponseError("malformed model reply")

    with pytest.raises(LLMError):
        raise LLMResponseError("malformed model reply")


# =============================================================== Anthropic path


async def test_anthropic_complete_returns_text_block_content(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC, model="claude-test")
    fake = _attach_anthropic(
        monkeypatch,
        client,
        _FakeAnthropicResponse(content=[_ThinkingBlock(), _TextBlock(text="the answer")]),
    )

    assert await client.complete("prompt text") == "the answer"
    assert len(fake.messages.calls) == 1


async def test_anthropic_complete_skips_non_text_sdk_blocks(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    """Real ``anthropic.types`` blocks: only ``TextBlock`` carries ``.text``."""
    client = make_client(provider=Provider.ANTHROPIC, model="claude-test")
    message = _real_anthropic_message(
        [
            ThinkingBlock(signature="sig", thinking="reasoning", type="thinking"),
            ToolUseBlock(id="tu_1", input={}, name="bash", type="tool_use"),
            TextBlock(text="real answer", type="text"),
        ]
    )
    _attach_anthropic(monkeypatch, client, _FakeAnthropicResponse(content=message.content))

    assert await client.complete("prompt text") == "real answer"


async def test_anthropic_complete_skips_blank_text_block(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC, model="claude-test")
    _attach_anthropic(
        monkeypatch,
        client,
        _FakeAnthropicResponse(
            content=[_TextBlock(text="   \n\t "), _TextBlock(text="real answer")]
        ),
    )

    assert await client.complete("prompt text") == "real answer"


async def test_anthropic_request_payload_is_exactly_the_documented_shape(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC, model="claude-payload", max_tokens=4242)
    fake = _attach_anthropic(monkeypatch, client, _ok_anthropic())

    await client.complete("write a poem")

    assert fake.messages.calls == [
        {
            "model": "claude-payload",
            "max_tokens": 4242,
            "system": DEFAULT_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": "write a poem"}],
        }
    ]


async def test_anthropic_sends_caller_supplied_system_prompt(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC, model="claude-test")
    fake = _attach_anthropic(monkeypatch, client, _ok_anthropic())

    await client.complete("prompt text", system="you are a linter")

    assert fake.messages.calls[0]["system"] == "you are a linter"


async def test_anthropic_complete_honours_explicit_model_override(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC, model="claude-default")
    fake = _attach_anthropic(monkeypatch, client, _ok_anthropic())

    await client.complete("prompt text", model="claude-override")

    assert fake.messages.calls[0]["model"] == "claude-override"


async def test_anthropic_complete_with_empty_content_raises_llm_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC, model="claude-test")
    _attach_anthropic(monkeypatch, client, _FakeAnthropicResponse(content=[]))

    with pytest.raises(LLMResponseError):
        await client.complete("prompt text")


async def test_anthropic_complete_with_thinking_only_content_raises_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    """A thinking-only reply must be an error, never a silent empty answer."""
    client = make_client(provider=Provider.ANTHROPIC, model="claude-test")
    message = _real_anthropic_message(
        [ThinkingBlock(signature="sig", thinking="reasoning", type="thinking")]
    )
    _attach_anthropic(monkeypatch, client, _FakeAnthropicResponse(content=message.content))

    with pytest.raises(LLMResponseError) as excinfo:
        await client.complete("prompt text")

    assert "ThinkingBlock" in str(excinfo.value)


async def test_anthropic_complete_with_non_list_content_raises_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC, model="claude-test")
    unusable = _FakeAnthropicResponse(content=cast(list[Any], "not a list of blocks"))
    _attach_anthropic(monkeypatch, client, unusable)

    with pytest.raises(LLMResponseError):
        await client.complete("prompt text")


async def test_anthropic_max_tokens_stop_reason_raises_llm_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    """Truncation must never be returned as if the answer were complete."""
    client = make_client(provider=Provider.ANTHROPIC, model="claude-test", max_tokens=17)
    _attach_anthropic(
        monkeypatch,
        client,
        _FakeAnthropicResponse(
            content=[_TextBlock(text="a truncated sentence")], stop_reason="max_tokens"
        ),
    )

    with pytest.raises(LLMResponseError) as excinfo:
        await client.complete("prompt text")

    assert "max_tokens" in str(excinfo.value)


# ================================================================== OpenAI path


async def test_openai_complete_returns_message_content(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OPENAI, model="gpt-test")
    fake = _attach_openai(
        monkeypatch,
        client,
        _FakeOpenAIResponse(
            choices=[_FakeOpenAIChoice(message=_FakeOpenAIMessage("the answer"))]
        ),
    )

    assert await client.complete("prompt text") == "the answer"
    assert len(fake.completions.calls) == 1


async def test_openai_request_payload_is_exactly_the_documented_shape(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OPENAI, model="gpt-payload", max_tokens=1234)
    fake = _attach_openai(monkeypatch, client, _ok_openai())

    await client.complete("write a poem")

    assert fake.completions.calls == [
        {
            "model": "gpt-payload",
            "max_tokens": 1234,
            "messages": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": "write a poem"},
            ],
        }
    ]


async def test_openai_sends_caller_supplied_system_prompt(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OPENAI, model="gpt-test")
    fake = _attach_openai(monkeypatch, client, _ok_openai())

    await client.complete("prompt text", system="you are a linter")

    assert fake.completions.calls[0]["messages"][0] == {
        "role": "system",
        "content": "you are a linter",
    }


async def test_openai_length_finish_reason_raises_llm_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OPENAI, model="gpt-test")
    _attach_openai(
        monkeypatch,
        client,
        _FakeOpenAIResponse(
            choices=[
                _FakeOpenAIChoice(
                    message=_FakeOpenAIMessage(content="a truncated sentence"),
                    finish_reason="length",
                )
            ]
        ),
    )

    with pytest.raises(LLMResponseError) as excinfo:
        await client.complete("prompt text")

    assert "length" in str(excinfo.value)


async def test_openai_empty_choices_raises_llm_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OPENAI, model="gpt-test")
    _attach_openai(monkeypatch, client, _FakeOpenAIResponse(choices=[]))

    with pytest.raises(LLMResponseError):
        await client.complete("prompt text")


async def test_openai_none_content_with_real_sdk_refusal_raises_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    """An empty reply used to collapse to ``""`` and then fail as a JSON parse error."""
    client = make_client(provider=Provider.OPENAI, model="gpt-test")
    completion = _real_openai_completion(
        ChatCompletionMessage(content=None, role="assistant", refusal="I cannot help"),
        finish_reason="content_filter",
    )
    _attach_openai(monkeypatch, client, _FakeOpenAIResponse(choices=completion.choices))

    with pytest.raises(LLMResponseError) as excinfo:
        await client.complete("prompt text")

    assert "I cannot help" in str(excinfo.value)


async def test_openai_none_content_with_tool_calls_raises_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OPENAI, model="gpt-test")
    _attach_openai(
        monkeypatch,
        client,
        _FakeOpenAIResponse(
            choices=[
                _FakeOpenAIChoice(
                    message=_FakeOpenAIMessage(
                        content=None, tool_calls=[object(), object()]
                    ),
                    finish_reason="tool_calls",
                )
            ]
        ),
    )

    with pytest.raises(LLMResponseError) as excinfo:
        await client.complete("prompt text")

    assert "tool_calls=2" in str(excinfo.value)


async def test_openai_blank_content_raises_llm_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OPENAI, model="gpt-test")
    _attach_openai(
        monkeypatch,
        client,
        _FakeOpenAIResponse(
            choices=[_FakeOpenAIChoice(message=_FakeOpenAIMessage(content="   "))]
        ),
    )

    with pytest.raises(LLMResponseError):
        await client.complete("prompt text")


# ================================================================= Ollama path


async def test_ollama_posts_expected_url_payload_and_timeout(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(
        provider=Provider.OLLAMA,
        model="qwen-test",
        max_tokens=321,
        request_timeout=7.5,
        ollama_base_url="http://ollama.invalid:11434",
    )
    body = '{"message": {"content": "the answer"}}'
    factory = _attach_ollama(
        monkeypatch, _FakeHttpxResponse(text=body, json_data={"message": {"content": "the answer"}})
    )

    assert await client.complete("write a poem") == "the answer"

    assert len(factory.instances) == 1
    built = factory.instances[0]
    assert built.init_kwargs == {"timeout": 7.5}
    assert built.calls == [
        (
            "http://ollama.invalid:11434/api/chat",
            {
                "json": {
                    "model": "qwen-test",
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": "write a poem"},
                    ],
                    "stream": False,
                    "options": {"num_predict": 321},
                }
            },
        )
    ]


async def test_ollama_base_url_trailing_slash_is_normalised(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(
        provider=Provider.OLLAMA,
        model="qwen-test",
        ollama_base_url="http://ollama.invalid:11434/",
    )
    factory = _attach_ollama(
        monkeypatch, _FakeHttpxResponse(json_data={"message": {"content": "ok"}})
    )

    await client.complete("prompt text")

    assert factory.instances[0].calls[0][0] == "http://ollama.invalid:11434/api/chat"


async def test_ollama_sends_caller_supplied_system_prompt(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OLLAMA, model="qwen-test")
    factory = _attach_ollama(
        monkeypatch, _FakeHttpxResponse(json_data={"message": {"content": "ok"}})
    )

    await client.complete("prompt text", system="you are a linter")

    messages = factory.instances[0].calls[0][1]["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "you are a linter"}


async def test_ollama_http_error_status_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OLLAMA, model="qwen-test")
    _attach_ollama(
        monkeypatch, _FakeHttpxResponse(status_code=500, text="internal server error")
    )

    with pytest.raises(LLMError) as excinfo:
        await client.complete("prompt text")

    assert not isinstance(excinfo.value, LLMResponseError)
    assert "500" in str(excinfo.value)


async def test_ollama_non_json_body_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OLLAMA, model="qwen-test")
    _attach_ollama(
        monkeypatch,
        _FakeHttpxResponse(text="<html>nope</html>", json_error=ValueError("not json")),
    )

    with pytest.raises(LLMError) as excinfo:
        await client.complete("prompt text")

    assert not isinstance(excinfo.value, LLMResponseError)


async def test_ollama_non_object_json_body_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OLLAMA, model="qwen-test")
    _attach_ollama(monkeypatch, _FakeHttpxResponse(text="[]", json_data=[]))

    with pytest.raises(LLMError) as excinfo:
        await client.complete("prompt text")

    assert not isinstance(excinfo.value, LLMResponseError)


async def test_ollama_transport_error_raises_llm_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OLLAMA, model="qwen-test")
    _attach_ollama(monkeypatch, _FakeHttpxResponse(), httpx.ConnectError("connection refused"))

    with pytest.raises(LLMError) as excinfo:
        await client.complete("prompt text")

    assert not isinstance(excinfo.value, LLMResponseError)
    assert "connection refused" in str(excinfo.value)


async def test_ollama_missing_message_content_raises_llm_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OLLAMA, model="qwen-test")
    body = '{"message": {"role": "assistant"}}'
    _attach_ollama(
        monkeypatch, _FakeHttpxResponse(text=body, json_data={"message": {"role": "assistant"}})
    )

    with pytest.raises(LLMResponseError):
        await client.complete("prompt text")


async def test_ollama_length_done_reason_raises_llm_response_error(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.OLLAMA, model="qwen-test", max_tokens=8)
    body = '{"done_reason": "length", "message": {"content": "cut off"}}'
    _attach_ollama(
        monkeypatch,
        _FakeHttpxResponse(
            text=body, json_data={"done_reason": "length", "message": {"content": "cut off"}}
        ),
    )

    with pytest.raises(LLMResponseError) as excinfo:
        await client.complete("prompt text")

    assert "length" in str(excinfo.value)


# =================================================================== lifecycle


async def test_aclose_closes_anthropic_client_exactly_once_when_called_twice(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC)
    fake = _attach_anthropic(monkeypatch, client, _ok_anthropic())

    await client.aclose()
    await client.aclose()

    assert fake.close_calls == 1


async def test_aclose_closes_both_provider_clients(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC)
    anthropic_fake = _attach_anthropic(monkeypatch, client, _ok_anthropic())
    openai_fake = _attach_openai(monkeypatch, client, _ok_openai())

    await client.aclose()
    await client.aclose()

    assert anthropic_fake.close_calls == 1
    assert openai_fake.close_calls == 1


async def test_aclose_resets_the_closed_provider_clients(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC)
    _attach_anthropic(monkeypatch, client, _ok_anthropic())
    _attach_openai(monkeypatch, client, _ok_openai())

    await client.aclose()

    assert client._anthropic is None
    assert client._openai is None


async def test_aclose_without_any_provider_client_is_a_noop(
    make_client: Callable[..., LLMClient],
) -> None:
    client = make_client()

    await client.aclose()
    await client.aclose()

    assert client._anthropic is None
    assert client._openai is None


async def test_async_context_manager_yields_the_client_and_closes_it(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC)
    fake = _attach_anthropic(monkeypatch, client, _ok_anthropic())

    async with client as entered:
        assert entered is client

    assert fake.close_calls == 1
    assert client._anthropic is None


async def test_async_context_manager_closes_the_client_on_exception(
    monkeypatch: pytest.MonkeyPatch, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client(provider=Provider.ANTHROPIC)
    fake = _attach_anthropic(monkeypatch, client, _ok_anthropic())

    with pytest.raises(RuntimeError, match="boom"):
        async with client:
            raise RuntimeError("boom")

    assert fake.close_calls == 1
    assert client._anthropic is None


# ============================================================ filesystem helpers


@pytest.mark.parametrize("method_name", ["read_file", "write_file", "list_files"])
def test_file_helpers_are_coroutine_functions(method_name: str) -> None:
    """``list_files`` became async; a sync call would hand back an un-awaited coroutine."""
    assert inspect.iscoroutinefunction(getattr(LLMClient, method_name))


async def test_write_then_read_round_trips_non_ascii_utf8(
    tmp_path: Path, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client()
    target = tmp_path / "unicode.txt"
    content = "héllo — wörld ✅ 日本語\nsecond line\n"

    await client.write_file(target, content)
    read_back = await client.read_file(target)

    assert read_back == content
    assert read_back.encode("utf-8") == content.encode("utf-8")


async def test_read_file_accepts_a_string_path(
    tmp_path: Path, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client()
    target = tmp_path / "plain.txt"
    target.write_text("plain content", encoding="utf-8")

    assert await client.read_file(str(target)) == "plain content"


async def test_write_file_creates_missing_parent_directories(
    tmp_path: Path, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client()
    target = tmp_path / "deeply" / "nested" / "pkg" / "module.py"

    await client.write_file(target, "x = 1\n")

    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "x = 1\n"


async def test_write_file_leaves_no_temporary_file_behind(
    tmp_path: Path, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client()
    target = tmp_path / "pkg" / "module.py"

    await client.write_file(target, "x = 1\n")

    assert [p.name for p in target.parent.iterdir()] == ["module.py"]


async def test_write_file_over_existing_file_replaces_content_completely(
    tmp_path: Path, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client()
    target = tmp_path / "module.py"
    target.write_text("y = 2\n" * 200, encoding="utf-8")

    await client.write_file(target, "z = 3\n")

    assert target.read_text(encoding="utf-8") == "z = 3\n"
    assert [p.name for p in tmp_path.iterdir()] == ["module.py"]


async def test_list_files_skips_noise_and_returns_root_relative_paths(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    make_client: Callable[..., LLMClient],
) -> None:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "README.md").write_text("readme", encoding="utf-8")
    (root / "pkg" / "mod.py").write_text("mod = 1\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "mod.cpython-310.pyc").write_bytes(b"\x00")
    (root / "pkg" / "stale.pyc").write_bytes(b"\x00")

    # An in-tree symlink is a legitimate file; one escaping the root is not.
    (root / "inside_link.py").symlink_to(root / "pkg" / "mod.py")
    outside = tmp_path_factory.mktemp("outside-repo")
    secret = outside / "secret.py"
    secret.write_text("secret = True\n", encoding="utf-8")
    (root / "escape_link.py").symlink_to(secret)

    client = make_client()

    assert await client.list_files(root) == ["README.md", "inside_link.py", "pkg/mod.py"]


async def test_list_files_excludes_symlink_escaping_the_root(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    make_client: Callable[..., LLMClient],
) -> None:
    """``relative_to`` used to raise ``ValueError`` on a path outside the tree."""
    root = tmp_path / "repo"
    root.mkdir()
    secret = tmp_path_factory.mktemp("outside-repo-escape") / "secret.py"
    secret.write_text("secret = True\n", encoding="utf-8")
    (root / "escape_link.py").symlink_to(secret)

    client = make_client()

    assert await client.list_files(root) == []


async def test_list_files_returns_empty_for_missing_directory(
    tmp_path: Path, make_client: Callable[..., LLMClient]
) -> None:
    client = make_client()

    assert await client.list_files(tmp_path / "does-not-exist") == []


async def test_list_files_on_empty_directory_returns_empty(
    tmp_path: Path, make_client: Callable[..., LLMClient]
) -> None:
    root = tmp_path / "empty"
    root.mkdir()

    client = make_client()

    assert await client.list_files(root) == []


async def test_list_files_results_are_sorted_and_relative(
    tmp_path: Path, make_client: Callable[..., LLMClient]
) -> None:
    root = tmp_path / "repo"
    (root / "b").mkdir(parents=True)
    for relative in ("z.py", "a.py", "b/y.py", "b/x.py"):
        (root / relative).write_text("", encoding="utf-8")

    client = make_client()
    found = await client.list_files(root)

    assert found == ["a.py", "b/x.py", "b/y.py", "z.py"]
    assert all(not name.startswith("/") for name in found)


async def test_list_files_accepts_a_string_path(
    tmp_path: Path, make_client: Callable[..., LLMClient]
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("", encoding="utf-8")

    client = make_client()

    assert await client.list_files(str(root)) == ["a.py"]