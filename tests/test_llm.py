"""Unit tests for ``furrow.llm``.

Protects the JSON salvage path (fences, prose, braces inside strings), the
"parse errors are never retried" retry policy, provider dispatch -- including the
Ollama provider, which must ride the OpenAI-compatible client at
``settings.ollama_base_url`` instead of raising -- and the filesystem helpers
(``list_files`` filtering, ``read_file``/``write_file`` round-tripping).

No network, no API key and no real SDK client is used: the provider SDKs are
replaced by recording fakes, and the exponential backoff is neutralised everywhere
except the single test that proves the real delay.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tenacity import wait_none

import furrow.llm as llm_mod
from furrow.config import Provider, Settings
from furrow.llm import (
    IGNORED_DIRS,
    LLMClient,
    LLMResponseError,
    extract_json_object,
    parse_json_object,
)


def _settings(**overrides: Any) -> Settings:
    """Build a throwaway ``Settings`` without touching the global singleton."""
    base = Settings(_env_file=None, workspace=Path("/nonexistent-furrow-workspace"))
    return base.model_copy(update=overrides) if overrides else base


# --------------------------------------------------------------------------- fakes


class FakeAnthropicSDK:
    """Stand-in for ``anthropic.AsyncAnthropic`` that records calls and replies."""

    def __init__(self, replies: Sequence[str | Exception]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.init_kwargs: dict[str, Any] = {}
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(content=[SimpleNamespace(text=reply)])


class FakeOpenAISDK:
    """Stand-in for ``openai.AsyncOpenAI`` (also used for the Ollama endpoint)."""

    def __init__(self, replies: Sequence[str | Exception]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.init_kwargs: dict[str, Any] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


def _use_anthropic(monkeypatch: pytest.MonkeyPatch, fake: FakeAnthropicSDK) -> None:
    def factory(**kwargs: Any) -> FakeAnthropicSDK:
        fake.init_kwargs = kwargs
        return fake

    monkeypatch.setattr(llm_mod, "AsyncAnthropic", factory)


def _use_openai(monkeypatch: pytest.MonkeyPatch, fake: FakeOpenAISDK) -> None:
    def factory(**kwargs: Any) -> FakeOpenAISDK:
        fake.init_kwargs = kwargs
        return fake

    monkeypatch.setattr(llm_mod, "AsyncOpenAI", factory)


def _forbid(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    def factory(**_kwargs: Any) -> Any:
        raise AssertionError(f"{name} must not be constructed for this provider")

    monkeypatch.setattr(llm_mod, name, factory)


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the tenacity wait strategy with ``wait_none`` so retries are instant."""
    monkeypatch.setattr(llm_mod, "wait_exponential", lambda **_kwargs: wait_none())


def _install(
    monkeypatch: pytest.MonkeyPatch, provider: Provider, replies: Sequence[str] = ()
) -> Any:
    """Wire both SDKs and return the fake the given provider will actually use."""
    anthropic, openai = FakeAnthropicSDK(replies), FakeOpenAISDK(replies)
    _use_anthropic(monkeypatch, anthropic)
    _use_openai(monkeypatch, openai)
    return anthropic if provider is Provider.ANTHROPIC else openai


# ------------------------------------------------------------------ extract_json_object


def test_extract_json_object_returns_plain_json() -> None:
    assert extract_json_object('{"a": 1, "b": "x"}') == '{"a": 1, "b": "x"}'


def test_extract_json_object_strips_json_code_fence() -> None:
    fenced = 'Here you go:\n```json\n{"tasks": [{"id": "1"}]}\n```\nHope that helps'
    assert json.loads(extract_json_object(fenced)) == {"tasks": [{"id": "1"}]}


def test_extract_json_object_accepts_bare_code_fence() -> None:
    assert json.loads(extract_json_object('```\n{"a": 1}\n```')) == {"a": 1}


def test_extract_json_object_finds_json_surrounded_by_prose() -> None:
    text = "Sure! I think this works:\n{\"a\": 1}\nLet me know if you need more."
    assert json.loads(extract_json_object(text)) == {"a": 1}


def test_extract_json_object_handles_braces_inside_strings() -> None:
    # A naive brace counter would stop at the "}" inside the string literal.
    assert json.loads(extract_json_object('prefix {"a": "}{"} suffix')) == {"a": "}{"}


def test_extract_json_object_handles_nested_objects_and_arrays() -> None:
    payload = '{"a": {"b": [1, {"c": [2, 3]}], "d": {"e": null}}}'
    assert json.loads(extract_json_object(payload)) == json.loads(payload)


def test_extract_json_object_handles_escaped_quotes_and_backslashes() -> None:
    payload = json.dumps({"a": 'he said "hi" \\ then left', "b": "C:\\Users\\dev"})
    assert json.loads(extract_json_object(payload)) == json.loads(payload)


def test_extract_json_object_ignores_surrounding_whitespace() -> None:
    assert json.loads(extract_json_object("\n\n   {\"a\": 1}   \n\t")) == {"a": 1}


def test_extract_json_object_skips_undecodable_braces_in_prose() -> None:
    # "{not json}" in the preamble must not shadow the real payload.
    text = "Format: {not json} ... actually: {\"a\": 1}"
    assert json.loads(extract_json_object(text)) == {"a": 1}


def test_extract_json_object_raises_when_no_object_present() -> None:
    with pytest.raises(LLMResponseError):
        extract_json_object("I'm sorry, I can't help with that request.")


# --------------------------------------------------------------------- parse_json_object


def test_parse_json_object_returns_a_dict() -> None:
    parsed = parse_json_object('noise {"passed": true, "failures": []} noise')
    assert parsed == {"passed": True, "failures": []}
    assert isinstance(parsed, dict)


def test_llm_response_error_is_a_value_error() -> None:
    assert issubclass(LLMResponseError, ValueError)


def test_parse_json_object_rejects_non_json_response() -> None:
    with pytest.raises(LLMResponseError):
        parse_json_object("no json at all")


def test_parse_json_object_rejects_json_that_is_not_an_object() -> None:
    with pytest.raises(LLMResponseError):
        parse_json_object("[1, 2, 3]")


def test_parse_json_object_rejects_json_scalar() -> None:
    with pytest.raises(LLMResponseError):
        parse_json_object('"just a string"')


def test_parse_json_object_error_contains_a_truncated_snippet() -> None:
    huge = "x" * 500 + " and no json"
    with pytest.raises(LLMResponseError) as excinfo:
        parse_json_object(huge)
    message = str(excinfo.value)
    assert "..." in message, "the snippet must be truncated with an ellipsis"
    assert len(message) < 400, f"error message is not truncated: {len(message)} chars"


# ------------------------------------------------------------------------ complete_json


async def test_complete_json_parses_the_completion_output(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAnthropicSDK(['{"ok": true}'])
    _use_anthropic(monkeypatch, fake)
    client = LLMClient(settings=_settings(anthropic_api_key="k"))
    assert await client.complete_json("hi") == {"ok": True}


async def test_complete_json_does_not_retry_parse_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    # llm_max_retries=3, but a ValueError/LLMResponseError must reach the caller once.
    fake = FakeAnthropicSDK(["I'm afraid I can't do that."])
    _use_anthropic(monkeypatch, fake)
    client = LLMClient(settings=_settings(anthropic_api_key="k", llm_max_retries=3))
    with pytest.raises(LLMResponseError):
        await client.complete_json("hi")
    assert len(fake.calls) == 1, "a parse failure must not be retried"


# ------------------------------------------------------------------------ retry policy


async def test_transient_failure_is_retried_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    fake = FakeAnthropicSDK([RuntimeError("429"), RuntimeError("503"), "recovered"])
    _use_anthropic(monkeypatch, fake)
    client = LLMClient(settings=_settings(anthropic_api_key="k", llm_max_retries=3))
    assert await client.complete("hi") == "recovered"
    assert len(fake.calls) == 3


async def test_retry_uses_a_real_exponential_backoff_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    # The one test that pays for the real tenacity wait: ~1s before the 2nd attempt.
    fake = FakeAnthropicSDK([RuntimeError("429"), "recovered"])
    _use_anthropic(monkeypatch, fake)
    client = LLMClient(settings=_settings(anthropic_api_key="k", llm_max_retries=2))
    started = time.monotonic()
    assert await client.complete("hi") == "recovered"
    elapsed = time.monotonic() - started
    assert elapsed >= 0.9, f"expected a backoff sleep before retrying, waited {elapsed:.2f}s"
    assert len(fake.calls) == 2


@pytest.mark.parametrize("attempts", [1, 2])
async def test_retries_stop_at_the_configured_limit(
    monkeypatch: pytest.MonkeyPatch, no_backoff: None, attempts: int
) -> None:
    fake = FakeAnthropicSDK([RuntimeError(f"boom {i}") for i in range(attempts + 5)])
    _use_anthropic(monkeypatch, fake)
    client = LLMClient(settings=_settings(anthropic_api_key="k", llm_max_retries=attempts))
    with pytest.raises(RuntimeError):
        await client.complete("hi")
    assert len(fake.calls) == attempts


@pytest.mark.parametrize("error", [ValueError("bad request"), LLMResponseError("bad json")])
async def test_value_errors_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, no_backoff: None, error: Exception
) -> None:
    fake = FakeAnthropicSDK([error, "never reached"])
    _use_anthropic(monkeypatch, fake)
    client = LLMClient(settings=_settings(anthropic_api_key="k", llm_max_retries=5))
    with pytest.raises(ValueError):
        await client.complete("hi")
    assert len(fake.calls) == 1


async def test_retry_policy_is_read_from_settings_at_call_time(
    monkeypatch: pytest.MonkeyPatch, no_backoff: None
) -> None:
    fake = FakeAnthropicSDK([RuntimeError("boom"), "second time lucky"])
    _use_anthropic(monkeypatch, fake)
    client = LLMClient(settings=_settings(anthropic_api_key="k", llm_max_retries=1))
    with pytest.raises(RuntimeError):
        await client.complete("hi")
    assert len(fake.calls) == 1
    client.settings = client.settings.model_copy(update={"llm_max_retries": 2})
    assert await client.complete("hi") == "second time lucky"
    assert len(fake.calls) == 2


# ------------------------------------------------------------------ provider dispatch


async def test_anthropic_provider_calls_the_anthropic_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAnthropicSDK(["hello from claude"])
    _use_anthropic(monkeypatch, fake)
    _forbid(monkeypatch, "AsyncOpenAI")
    client = LLMClient(settings=_settings(provider=Provider.ANTHROPIC, anthropic_api_key="k"))
    assert await client.complete("hi", system="be terse") == "hello from claude"
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["system"] == "be terse"


async def test_openai_provider_calls_the_openai_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOpenAISDK(["hello from gpt"])
    _use_openai(monkeypatch, fake)
    _forbid(monkeypatch, "AsyncAnthropic")
    client = LLMClient(settings=_settings(provider=Provider.OPENAI, openai_api_key="k"))
    assert await client.complete("hi", system="be terse") == "hello from gpt"
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


async def test_ollama_provider_uses_the_openai_compatible_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: Provider.OLLAMA used to raise ValueError instead of being routed
    # through an OpenAI-compatible client pointed at settings.ollama_base_url.
    fake = FakeOpenAISDK(["hello from llama"])
    _use_openai(monkeypatch, fake)
    _forbid(monkeypatch, "AsyncAnthropic")
    settings = _settings(
        provider=Provider.OLLAMA,
        openai_api_key=None,
        ollama_base_url="http://localhost:1234",
    )
    client = LLMClient(settings=settings)
    assert await client.complete("hi") == "hello from llama"
    assert fake.init_kwargs["base_url"] == "http://localhost:1234"
    assert fake.init_kwargs["api_key"]  # ollama needs no real key


async def test_unknown_provider_still_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid(monkeypatch, "AsyncAnthropic")
    _forbid(monkeypatch, "AsyncOpenAI")
    # model_copy skips validation, so this is how an unrecognised provider arrives.
    settings = _settings(provider="bedrock")  # type: ignore[arg-type]
    client = LLMClient(settings=settings)
    with pytest.raises(ValueError):
        await client.complete("hi")


@pytest.mark.parametrize("provider", list(Provider))
async def test_max_tokens_setting_is_forwarded_to_the_sdk(
    monkeypatch: pytest.MonkeyPatch, provider: Provider
) -> None:
    fake = _install(monkeypatch, provider, ["ok"])
    client = LLMClient(
        settings=_settings(
            provider=provider,
            anthropic_api_key="k",
            openai_api_key="k",
            max_tokens=12345,
        )
    )
    assert await client.complete("hi") == "ok"
    assert fake.calls[0]["max_tokens"] == 12345


@pytest.mark.parametrize("provider", list(Provider))
async def test_request_timeout_is_forwarded_to_the_sdk_constructor(
    monkeypatch: pytest.MonkeyPatch, provider: Provider
) -> None:
    fake = _install(monkeypatch, provider, ["ok"])
    client = LLMClient(
        settings=_settings(
            provider=provider,
            anthropic_api_key="k",
            openai_api_key="k",
            request_timeout=12.5,
        )
    )
    await client.complete("hi")
    assert fake.init_kwargs["timeout"] == 12.5


async def test_model_argument_overrides_the_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAnthropicSDK(["ok"])
    _use_anthropic(monkeypatch, fake)
    client = LLMClient(settings=_settings(anthropic_api_key="k", model="default-model"))
    await client.complete("hi", model="explicit-model")
    assert fake.calls[0]["model"] == "explicit-model"


async def test_missing_api_key_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(settings=_settings(provider=Provider.ANTHROPIC, anthropic_api_key=None))
    with pytest.raises(ValueError):
        _ = client.anthropic


# ----------------------------------------------------------- settings resolution


def test_settings_default_argument_is_none_not_the_import_time_singleton() -> None:
    # Regression: `def __init__(self, settings: Settings = settings)` froze the
    # singleton at import time, so later overrides were ignored.
    default = inspect.signature(LLMClient.__init__).parameters["settings"].default
    assert default is None


def test_llm_client_resolves_the_settings_singleton_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = _settings(max_tokens=11)
    monkeypatch.setattr(llm_mod, "_DEFAULT_SETTINGS", sentinel)
    assert LLMClient().settings is sentinel
    assert LLMClient(settings=None).settings is sentinel
    explicit = _settings(max_tokens=22)
    assert LLMClient(settings=explicit).settings is explicit
    assert LLMClient(settings=explicit).settings is not sentinel


def test_default_ignored_dirs_are_still_declared() -> None:
    assert {".git", "__pycache__", "node_modules", ".venv"} <= IGNORED_DIRS


# ------------------------------------------------------------------------ file helpers


def _tree(root: Path) -> None:
    for name in ("a.py", "b/c.py", "d/e/f.py", ".hidden.py", ".hidden_dir/g.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    for ignored in (".git", "__pycache__", "node_modules", ".venv"):
        path = root / ignored / "junk.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("junk", encoding="utf-8")


def test_list_files_defaults_to_the_settings_workspace(tmp_path: Path) -> None:
    _tree(tmp_path)
    client = LLMClient(settings=_settings(workspace=tmp_path))
    assert client.list_files() == ["a.py", "b/c.py", "d/e/f.py"]


def test_list_files_skips_hidden_entries_and_ignored_dirs(tmp_path: Path) -> None:
    _tree(tmp_path)
    client = LLMClient(settings=_settings(workspace=tmp_path))
    listed = client.list_files()
    assert listed == ["a.py", "b/c.py", "d/e/f.py"]
    assert not any(".hidden" in name for name in listed)
    for ignored in (".git", "__pycache__", "node_modules", ".venv"):
        assert not any(part == ignored for name in listed for part in name.split("/"))


def test_list_files_returns_sorted_relative_posix_paths(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "zeta.txt").write_text("z", encoding="utf-8")
    (tmp_path / "pkg" / "mod.py").write_text("m", encoding="utf-8")
    client = LLMClient(settings=_settings(workspace=tmp_path))
    listed = client.list_files()
    assert listed == sorted(listed)
    assert listed == ["pkg/mod.py", "zeta.txt"]
    assert all("\\" not in name for name in listed)


def test_list_files_accepts_an_explicit_directory(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.py").write_text("x", encoding="utf-8")
    client = LLMClient(settings=_settings(workspace=tmp_path))
    assert client.list_files(tmp_path / "sub") == ["x.py"]


def test_list_files_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    client = LLMClient(settings=_settings(workspace=tmp_path))
    assert client.list_files(tmp_path / "nope") == []


def test_list_files_returns_empty_for_a_file_path(tmp_path: Path) -> None:
    target = tmp_path / "a-file"
    target.write_text("x", encoding="utf-8")
    client = LLMClient(settings=_settings(workspace=tmp_path))
    assert client.list_files(target) == []


async def test_write_file_then_read_file_round_trips(tmp_path: Path) -> None:
    client = LLMClient(settings=_settings(workspace=tmp_path))
    target = tmp_path / "notes.txt"
    await client.write_file(target, "héllo\nwörld ✅\n")
    assert await client.read_file(target) == "héllo\nwörld ✅\n"


async def test_write_file_creates_missing_parent_directories(tmp_path: Path) -> None:
    client = LLMClient(settings=_settings(workspace=tmp_path))
    target = tmp_path / "deep" / "nested" / "tree" / "file.txt"
    await client.write_file(target, "content")
    assert target.is_file()
    assert await client.read_file(target) == "content"


async def test_write_file_accepts_a_string_path(tmp_path: Path) -> None:
    client = LLMClient(settings=_settings(workspace=tmp_path))
    await client.write_file(str(tmp_path / "s" / "t.txt"), "ok")
    assert await client.read_file(str(tmp_path / "s" / "t.txt")) == "ok"