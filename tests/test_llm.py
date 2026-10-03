"""Tests for :mod:`furrow.llm`: JSON extraction, provider dispatch, retries, files.

Everything here is offline: the provider SDKs and ``httpx`` are replaced with local
stubs so no socket is ever opened and no API key is ever needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from tenacity import AsyncRetrying

from furrow.config import Provider, Settings
from furrow.config import settings as global_settings
from furrow.llm import (
    DEFAULT_SYSTEM_PROMPT,
    RETRY_ATTEMPTS,
    SNIPPET_LIMIT,
    JSONParseError,
    LLMClient,
    LLMError,
    MissingAPIKeyError,
    UnsupportedProviderError,
    extract_json,
)

# --------------------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ambient provider keys / ``FURROW_*`` overrides out of the tests."""
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name in [item for item in os.environ if item.startswith("FURROW_")]:
        monkeypatch.delenv(name, raising=False)


def _settings(**overrides: Any) -> Settings:
    """An isolated ``Settings`` instance (never the global singleton)."""
    return Settings(_env_file=None, **overrides)


def _client(**overrides: Any) -> LLMClient:
    """An ``LLMClient`` bound to an isolated ``Settings``."""
    values: dict[str, Any] = {"provider": "anthropic", "anthropic_api_key": "sk-test-key"}
    values.update(overrides)
    return LLMClient(_settings(**values))


class _InstantAsyncRetrying(AsyncRetrying):
    """``AsyncRetrying`` with the backoff removed so retries do not really sleep."""

    def __init__(self, **kwargs: Any) -> None:
        async def _no_sleep(_seconds: float) -> None:
            return None

        kwargs["sleep"] = _no_sleep
        super().__init__(**kwargs)


class _FakeMessages:
    def __init__(self, response: Any, calls: list[dict[str, Any]]) -> None:
        self._response = response
        self.calls = calls

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


class _FakeAnthropic:
    """Stands in for ``anthropic.AsyncAnthropic``."""

    def __init__(self, response: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = _FakeMessages(response, self.calls)


class _FakeOpenAI:
    """Stands in for ``openai.AsyncOpenAI``."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_checked = False

    def raise_for_status(self) -> None:
        self.status_checked = True

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttpx:
    """Stands in for ``httpx.AsyncClient``; records ``post`` calls, does no I/O."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[_FakeResponse] = []

    async def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
        self.calls.append((url, json))
        response = _FakeResponse(self._payload)
        self.responses.append(response)
        return response


class _ClosableStub:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _ExplodingStub:
    def close(self) -> None:
        raise RuntimeError("close failed on purpose")


def _text_block(text: str) -> Any:
    return type("Block", (), {"type": "text", "text": text})()


def _thinking_block() -> Any:
    return type("Block", (), {"type": "thinking", "thinking": "hmm"})()


# ------------------------------------------------------------------- extract_json


def test_extract_json_bare_object() -> None:
    assert extract_json('{"tasks": ["a"], "n": 2}') == {"tasks": ["a"], "n": 2}


def test_extract_json_fenced_json_block() -> None:
    text = '```json\n{"ok": true}\n```'
    assert extract_json(text) == {"ok": True}


def test_extract_json_fence_wrapped_in_prose() -> None:
    text = (
        "Sure! Here is the plan you asked for:\n"
        "```json\n"
        '{"tasks": [{"id": "1"}]}\n'
        "```\n"
        "Let me know if you want changes."
    )
    assert extract_json(text) == {"tasks": [{"id": "1"}]}


def test_extract_json_top_level_array() -> None:
    assert extract_json('```\n[{"id": "1"}, {"id": "2"}]\n```') == [
        {"id": "1"},
        {"id": "2"},
    ]


def test_extract_json_ignores_braces_and_escapes_inside_strings() -> None:
    payload = {
        "code": 'def f():\n    return {"a": "}{", "b": 1}',
        "note": 'a quoted \\" brace {',
        "nested": {"deep": {"deeper": [1, 2, 3]}},
    }
    text = "Here you go:\n" + json.dumps(payload) + "\nDone."
    assert extract_json(text) == payload


def test_extract_json_takes_the_first_balanced_span() -> None:
    assert extract_json('noise {"a": 1} trailing {"b": 2}') == {"a": 1}


@pytest.mark.parametrize("text", ["", "   ", "\n\t  \n"])
def test_extract_json_rejects_blank_input(text: str) -> None:
    with pytest.raises(JSONParseError):
        extract_json(text)


def test_extract_json_error_message_is_truncated() -> None:
    tail_marker = "UNIQUE_TAIL_MARKER_MUST_NOT_LEAK"
    text = "no json here " * 40 + tail_marker
    assert len(text.strip()) > SNIPPET_LIMIT
    with pytest.raises(JSONParseError) as excinfo:
        extract_json(text)
    message = str(excinfo.value)
    assert message.startswith("No JSON object or array found in model output:")
    assert len(message) < 400
    # The first SNIPPET_LIMIT characters are echoed, the rest is dropped.
    assert text.strip()[:SNIPPET_LIMIT] in message
    assert tail_marker not in message
    assert text.strip() not in message


def test_extract_json_rejects_non_string_input() -> None:
    with pytest.raises(JSONParseError) as excinfo:
        extract_json({"already": "parsed"})  # type: ignore[arg-type]
    assert "dict" in str(excinfo.value)


# --------------------------------------------------------------- client construction


def test_client_without_settings_uses_global_singleton() -> None:
    client = LLMClient()
    assert client.settings is global_settings
    assert client.provider == global_settings.provider


def test_client_with_custom_settings_does_not_mutate_singleton(tmp_path: Path) -> None:
    before = global_settings.model_dump()
    client = LLMClient(_settings(workspace=tmp_path, model="custom-model"))
    assert client.settings.model == "custom-model"
    assert client.settings.workspace == tmp_path
    assert global_settings is not client.settings
    assert global_settings.model_dump() == before


def test_client_does_not_share_provider_caches() -> None:
    first = _client()
    second = _client()
    assert first._anthropic is None
    assert first._openai is None
    assert first._httpx is None
    assert second is not first


def test_token_counters_start_at_zero() -> None:
    client = _client()
    assert client.input_tokens == 0
    assert client.output_tokens == 0


# ------------------------------------------------------------------- provider dispatch


async def test_unknown_provider_raises_unsupported_provider() -> None:
    class _UnknownProviderClient(LLMClient):
        @property
        def provider(self) -> Any:
            return "gemini-1.5-pro"

    client = _UnknownProviderClient(_settings(anthropic_api_key="sk-test-key"))
    with pytest.raises(UnsupportedProviderError) as excinfo:
        await client.complete("hello")
    message = str(excinfo.value)
    assert "gemini-1.5-pro" in message
    assert "anthropic, openai, ollama" in message
    assert isinstance(excinfo.value, LLMError)


async def test_missing_anthropic_key_names_env_var_and_leaks_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret-value")
    client = _client(provider="anthropic", anthropic_api_key=None)
    with pytest.raises(MissingAPIKeyError) as excinfo:
        await client.complete("hello")
    message = str(excinfo.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "FURROW_ANTHROPIC_API_KEY" in message
    assert "sk-openai-secret-value" not in message


async def test_missing_openai_key_names_env_var_and_leaks_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-secret-value")
    client = _client(provider="openai", openai_api_key=None, anthropic_api_key="sk-leak-me")
    with pytest.raises(MissingAPIKeyError) as excinfo:
        await client.complete("hello")
    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert "FURROW_OPENAI_API_KEY" in message
    assert "sk-leak-me" not in message
    assert "sk-anthropic-secret-value" not in message


def test_provider_property_returns_enum_member() -> None:
    assert _client(provider="ollama").provider is Provider.OLLAMA


# ----------------------------------------------------------------------------- retries


async def test_complete_retries_transient_failure_up_to_the_attempt_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("furrow.llm.AsyncRetrying", _InstantAsyncRetrying)
    client = _client()
    attempts: list[int] = []

    async def _boom(*args: Any, **kwargs: Any) -> str:
        attempts.append(1)
        raise RuntimeError("transient provider error")

    monkeypatch.setattr(client, "_complete_anthropic", _boom)
    with pytest.raises(RuntimeError, match="transient provider error"):
        await client.complete("hello")
    assert len(attempts) == RETRY_ATTEMPTS == 3


async def test_complete_retries_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("furrow.llm.AsyncRetrying", _InstantAsyncRetrying)
    client = _client()
    attempts: list[int] = []

    async def _flaky(*args: Any, **kwargs: Any) -> str:
        attempts.append(1)
        if len(attempts) < 2:
            raise ConnectionError("reset by peer")
        return "recovered"

    monkeypatch.setattr(client, "_complete_anthropic", _flaky)
    assert await client.complete("hello") == "recovered"
    assert len(attempts) == 2


async def test_complete_does_not_retry_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("furrow.llm.AsyncRetrying", _InstantAsyncRetrying)
    client = _client()
    attempts: list[int] = []

    async def _no_key(*args: Any, **kwargs: Any) -> str:
        attempts.append(1)
        raise MissingAPIKeyError("ANTHROPIC_API_KEY is not set.")

    monkeypatch.setattr(client, "_complete_anthropic", _no_key)
    with pytest.raises(MissingAPIKeyError):
        await client.complete("hello")
    assert len(attempts) == 1


async def test_complete_does_not_retry_a_deterministic_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic ``LLMError`` fails fast; a transient one still retries.

    Every ``LLMError`` the module raises (empty Anthropic response, no OpenAI
    choices, no Ollama content, bad key, unparseable reply) is deterministic, so
    retrying it only burns attempts and the real ``wait_exponential`` backoff.
    Transient SDK/network failures are *not* ``LLMError`` and keep the full
    retry budget.
    """
    monkeypatch.setattr("furrow.llm.AsyncRetrying", _InstantAsyncRetrying)
    client = _client()
    deterministic: list[int] = []
    transient: list[int] = []

    async def _empty(*args: Any, **kwargs: Any) -> str:
        deterministic.append(1)
        raise LLMError("Anthropic returned an empty response")

    async def _boom(*args: Any, **kwargs: Any) -> str:
        transient.append(1)
        raise RuntimeError("transient provider error")

    monkeypatch.setattr(client, "_complete_anthropic", _empty)
    with pytest.raises(LLMError, match="empty response"):
        await client.complete("hello")
    assert len(deterministic) == 1

    monkeypatch.setattr(client, "_complete_anthropic", _boom)
    with pytest.raises(RuntimeError, match="transient provider error"):
        await client.complete("hello")
    assert len(transient) == RETRY_ATTEMPTS == 3


# -------------------------------------------------------------------------- complete_json


async def test_complete_json_parses_fenced_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    prompts: list[str] = []

    async def _reply(prompt: str, system: str = "", model: str | None = None) -> str:
        prompts.append(prompt)
        return 'Sure thing:\n```json\n{"passed": true, "failures": []}\n```'

    monkeypatch.setattr(client, "complete", _reply)
    assert await client.complete_json("go") == {"passed": True, "failures": []}
    assert prompts == ["go"]


async def test_complete_json_repairs_once_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    prompts: list[str] = []

    async def _reply(prompt: str, system: str = "", model: str | None = None) -> str:
        prompts.append(prompt)
        return "sorry, no json" if len(prompts) == 1 else '{"ok": true}'

    monkeypatch.setattr(client, "complete", _reply)
    assert await client.complete_json("go") == {"ok": True}
    assert len(prompts) == 2
    assert prompts[1].startswith("go")
    assert "could not be parsed as JSON" in prompts[1]


async def test_complete_json_raises_after_failed_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client()
    prompts: list[str] = []

    async def _reply(prompt: str, system: str = "", model: str | None = None) -> str:
        prompts.append(prompt)
        return f"still not json (attempt {len(prompts)})"

    monkeypatch.setattr(client, "complete", _reply)
    with pytest.raises(JSONParseError) as excinfo:
        await client.complete_json("go")
    message = str(excinfo.value)
    assert len(prompts) == 2
    assert "2 attempt(s)" in message
    assert "still not json (attempt 2)" in message


async def test_complete_json_without_repairs_makes_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _client()
    prompts: list[str] = []

    async def _reply(prompt: str, system: str = "", model: str | None = None) -> str:
        prompts.append(prompt)
        return "nope"

    monkeypatch.setattr(client, "complete", _reply)
    with pytest.raises(JSONParseError):
        await client.complete_json("go", repair_attempts=0)
    assert len(prompts) == 1


# ------------------------------------------------------------------------- anthropic


async def test_anthropic_path_skips_non_text_blocks_and_records_usage() -> None:
    usage = type("Usage", (), {"input_tokens": 11, "output_tokens": 7})()
    response = type(
        "Response",
        (),
        {
            "content": [_thinking_block(), _text_block("hello "), _text_block("world")],
            "usage": usage,
        },
    )()
    fake = _FakeAnthropic(response)
    client = _client()
    client._anthropic = fake
    assert await client.complete("hi") == "hello world"
    assert client.input_tokens == 11
    assert client.output_tokens == 7


async def test_anthropic_path_raises_on_text_less_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("furrow.llm.AsyncRetrying", _InstantAsyncRetrying)
    response = type("Response", (), {"content": [_thinking_block()], "usage": None})()
    client = _client()
    client._anthropic = _FakeAnthropic(response)
    with pytest.raises(LLMError, match="empty response"):
        await client.complete("hi")


async def test_anthropic_path_sends_documented_request_kwargs() -> None:
    usage = type("Usage", (), {"input_tokens": 1, "output_tokens": 2})()
    response = type("Response", (), {"content": [_text_block("ok")], "usage": usage})()
    fake = _FakeAnthropic(response)
    client = _client(model="claude-x", temperature=0.25, max_tokens=512)
    client._anthropic = fake
    assert await client.complete("hi", system="be terse") == "ok"
    assert fake.calls == [
        {
            "model": "claude-x",
            "max_tokens": 512,
            "temperature": 0.25,
            "system": "be terse",
            "messages": [{"role": "user", "content": "hi"}],
        }
    ]


async def test_anthropic_path_defaults_the_system_prompt() -> None:
    response = type("Response", (), {"content": [_text_block("ok")], "usage": None})()
    fake = _FakeAnthropic(response)
    client = _client()
    client._anthropic = fake
    await client.complete("hi")
    assert fake.calls[0]["system"] == DEFAULT_SYSTEM_PROMPT


async def test_anthropic_model_override_wins_over_settings() -> None:
    response = type("Response", (), {"content": [_text_block("ok")], "usage": None})()
    fake = _FakeAnthropic(response)
    client = _client(model="claude-settings")
    client._anthropic = fake
    await client.complete("hi", model="claude-explicit")
    assert fake.calls[0]["model"] == "claude-explicit"


# ---------------------------------------------------------------------------- openai


async def test_openai_path_returns_first_choice_content() -> None:
    usage = type("Usage", (), {"prompt_tokens": 3, "completion_tokens": 4})()
    message = type("Msg", (), {"content": "from openai"})()
    choice = type("Choice", (), {"message": message})()
    response = type("Response", (), {"choices": [choice], "usage": usage})()
    fake = _FakeOpenAI(response)
    client = _client(provider="openai", openai_api_key="sk-openai-test")
    client._openai = fake
    assert await client.complete("hi") == "from openai"
    assert client.input_tokens == 3
    assert client.output_tokens == 4


async def test_openai_path_raises_without_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("furrow.llm.AsyncRetrying", _InstantAsyncRetrying)
    response = type("Response", (), {"choices": [], "usage": None})()
    client = _client(provider="openai", openai_api_key="sk-openai-test")
    client._openai = _FakeOpenAI(response)
    with pytest.raises(LLMError, match="no choices"):
        await client.complete("hi")


# ----------------------------------------------------------------------------- ollama


async def test_ollama_path_posts_documented_payload_and_returns_content() -> None:
    fake = _FakeHttpx(
        {
            "message": {"role": "assistant", "content": "ollama says hi"},
            "prompt_eval_count": 5,
            "eval_count": 6,
        }
    )
    client = _client(
        provider="ollama",
        ollama_base_url="http://ollama.test:11434/",
        ollama_model="llama3.1",
        temperature=0.3,
        max_tokens=512,
    )
    client._httpx = fake
    assert await client.complete("hi", system="be terse") == "ollama says hi"
    assert len(fake.calls) == 1
    url, payload = fake.calls[0]
    assert url == "http://ollama.test:11434/api/chat"
    assert payload == {
        "model": "llama3.1",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 512},
    }
    assert fake.responses[0].status_checked is True
    assert client.input_tokens == 5
    assert client.output_tokens == 6


async def test_ollama_path_rejects_blank_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("furrow.llm.AsyncRetrying", _InstantAsyncRetrying)
    client = _client(provider="ollama")
    client._httpx = _FakeHttpx({"message": {"content": "   "}})
    with pytest.raises(LLMError, match="no content"):
        await client.complete("hi")


async def test_ollama_model_defaults_to_ollama_model_setting() -> None:
    fake = _FakeHttpx({"message": {"content": "ok"}})
    client = _client(provider="ollama", ollama_model="qwen2.5")
    client._httpx = fake
    await client.complete("hi")
    assert fake.calls[0][1]["model"] == "qwen2.5"


# ------------------------------------------------------------------------ token usage


async def test_token_counters_accumulate_across_completions() -> None:
    response = type("Response", (), {"content": [_text_block("ok")], "usage": None})()
    client = _client()
    client._anthropic = _FakeAnthropic(response)
    client._record_usage(3, 4)
    client._record_usage("5", 6)
    client._record_usage(None, True)
    assert (client.input_tokens, client.output_tokens) == (8, 10)


def test_record_usage_ignores_unusable_values() -> None:
    client = _client()
    client._record_usage("not-a-number", object())
    client._record_usage(None, None)
    assert (client.input_tokens, client.output_tokens) == (0, 0)


# ---------------------------------------------------------------------------- files


async def test_write_then_read_file_round_trip(tmp_path: Path) -> None:
    client = _client()
    target = tmp_path / "nested" / "deep" / "notes.md"
    await client.write_file(target, "hello furrow")
    assert target.read_text(encoding="utf-8") == "hello furrow"
    assert await client.read_file(target) == "hello furrow"
    assert await client.read_file(str(target)) == "hello furrow"


async def test_write_file_overwrites_existing_content(tmp_path: Path) -> None:
    client = _client()
    target = tmp_path / "a.txt"
    await client.write_file(target, "first")
    await client.write_file(target, "second")
    assert await client.read_file(target) == "second"


def _build_workspace(root: Path) -> None:
    (root / "src").mkdir()
    (root / "README.md").write_text("readme", encoding="utf-8")
    (root / "src" / "app.py").write_text("app", encoding="utf-8")
    (root / "src" / "main.py").write_text("main", encoding="utf-8")
    for noise in (".git", "__pycache__", "node_modules", ".venv", ".hidden", "build", "dist"):
        directory = root / noise
        directory.mkdir()
        (directory / "junk.py").write_text("junk", encoding="utf-8")
    (root / ".gitignore").write_text("*.pyc", encoding="utf-8")
    (root / "src" / ".env").write_text("SECRET=1", encoding="utf-8")
    link = root / "src" / "link.py"
    try:
        link.symlink_to(root / "src" / "app.py")
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pass


def test_list_files_returns_sorted_relative_paths(tmp_path: Path) -> None:
    _build_workspace(tmp_path)
    client = _client()
    assert client.list_files(tmp_path) == ["README.md", "src/app.py", "src/main.py"]


def test_list_files_excludes_noise_and_dot_entries(tmp_path: Path) -> None:
    _build_workspace(tmp_path)
    client = _client()
    found = client.list_files(tmp_path)
    assert not [name for name in found if name.startswith(".")]
    for noise in ("__pycache__", "node_modules", ".venv", "build", "dist", "link.py"):
        assert not [name for name in found if noise in name]


def test_list_files_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    client = _client()
    assert client.list_files(tmp_path / "does-not-exist") == []
    assert client.list_files(tmp_path / "a-file.txt") == []


def test_list_files_ignores_symlinked_directories(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "kept.py").write_text("kept", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    try:
        (root / "linked").symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
        pytest.skip("symlinks unavailable")
    assert _client().list_files(root) == []


# -------------------------------------------------------------------------- lifecycle


async def test_close_is_idempotent() -> None:
    client = _client()
    anthropic_stub = _ClosableStub()
    httpx_stub = _ClosableStub()
    client._anthropic = anthropic_stub
    client._httpx = httpx_stub
    await client.close()
    await client.close()
    assert anthropic_stub.close_calls == 1
    assert httpx_stub.close_calls == 1
    assert client._anthropic is None
    assert client._httpx is None


async def test_close_never_raises_on_provider_errors() -> None:
    client = _client()
    client._httpx = _ExplodingStub()
    await client.close()
    await client.close()


async def test_async_context_manager_closes_the_client() -> None:
    client = _client()
    stub = _ClosableStub()
    client._httpx = stub
    async with client as entered:
        assert entered is client
    assert stub.close_calls == 1