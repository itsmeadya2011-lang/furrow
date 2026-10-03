"""Tests for :mod:`furrow.llm`.

The client is the boundary between the orchestrator and paid network calls, so
everything here runs against fake SDK objects and a fake HTTP transport:
provider dispatch, Anthropic content-block extraction, retry classification,
and the file helpers. No test touches the network or sleeps.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import tenacity
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock

from furrow import llm
from furrow.config import Provider, Settings
from furrow.llm import LLMClient


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry instantly: tenacity waits are patched out so tests never sleep."""
    monkeypatch.setattr(llm, "wait_exponential_jitter", lambda **kw: tenacity.wait.wait_none())


class AuthError(Exception):
    """A 401-shaped SDK failure: real, but never worth retrying."""

    status_code = 401


def _fake_anthropic(blocks: list[Any], captured: dict[str, Any]) -> Any:
    """An object shaped like ``AsyncAnthropic`` that records the create() kwargs."""

    class _Messages:
        async def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(content=blocks)

    return SimpleNamespace(messages=_Messages())


# --------------------------------------------------------------------------- settings


def test_settings_parameter_does_not_shadow_module_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: the __init__ parameter used to shadow the module global.
    assert LLMClient().settings is llm.global_settings
    assert LLMClient(None).settings is llm.global_settings

    custom = Settings(model="custom-model")
    assert LLMClient(settings=custom).settings is custom

    replacement = Settings(model="replaced-global")
    monkeypatch.setattr(llm, "global_settings", replacement)
    assert LLMClient().settings is replacement


async def test_complete_requires_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        await client.complete("hi")


# --------------------------------------------------------------------------- dispatch


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (Provider.ANTHROPIC, "_complete_anthropic"),
        (Provider.OPENAI, "_complete_openai"),
        (Provider.OLLAMA, "_complete_ollama"),
    ],
)
async def test_complete_dispatches_on_provider(
    monkeypatch: pytest.MonkeyPatch, provider: Provider, expected: str
) -> None:
    calls: list[tuple[str, str, str]] = []

    for name in ("_complete_anthropic", "_complete_openai", "_complete_ollama"):

        async def record(
            self: LLMClient, prompt: str, system: str, model: str, name: str = name
        ) -> str:
            calls.append((name, prompt, model))
            return name

        monkeypatch.setattr(LLMClient, name, record)

    client = LLMClient(Settings(provider=provider, model="m-default", ollama_model="llama-x"))
    assert await client.complete("p", system="s") == expected
    assert calls == [(expected, "p", "llama-x" if provider is Provider.OLLAMA else "m-default")]


async def test_ollama_model_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    async def fake(self: LLMClient, prompt: str, system: str, model: str) -> str:
        seen.append(model)
        return "ok"

    monkeypatch.setattr(LLMClient, "_complete_ollama", fake)
    empty = LLMClient(Settings(provider=Provider.OLLAMA, ollama_model="", model="general"))
    assert await empty.complete("p") == "ok"
    # explicit argument beats both, an unset ollama_model falls back to model
    assert await empty.complete("p", model="explicit") == "ok"
    assert seen == ["general", "explicit"]


# --------------------------------------------------------------------------- anthropic


_CONTENT_CASES = [
    pytest.param([], "", id="empty"),
    pytest.param([TextBlock(text="only", type="text")], "only", id="single"),
    pytest.param(
        [TextBlock(text="a", type="text"), TextBlock(text="b", type="text")],
        "ab",
        id="multiple",
    ),
    pytest.param(
        [
            TextBlock(text="Hello ", type="text"),
            ThinkingBlock(thinking="secret", signature="sig", type="thinking"),
            ToolUseBlock(id="t1", name="bash", input={}, type="tool_use"),
            TextBlock(text="world", type="text"),
        ],
        "Hello world",
        id="non-text-blocks-skipped",
    ),
    pytest.param(
        [SimpleNamespace(text="fake "), SimpleNamespace(other=1), SimpleNamespace(text="dup")],
        "fake dup",
        id="duck-typed",
    ),
]


@pytest.mark.parametrize(("blocks", "expected"), _CONTENT_CASES)
async def test_complete_anthropic_returns_only_text(
    monkeypatch: pytest.MonkeyPatch, blocks: list[Any], expected: str
) -> None:
    captured: dict[str, Any] = {}
    client = LLMClient(Settings(anthropic_api_key="test-key", model="m-1"))
    monkeypatch.setattr(LLMClient, "anthropic", _fake_anthropic(blocks, captured))

    assert await client.complete("hi", system="sys") == expected
    assert captured["model"] == "m-1"
    assert captured["max_tokens"] == client.settings.llm_max_tokens
    assert captured["system"] == "sys"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


async def test_complete_anthropic_defaults_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    client = LLMClient(Settings(anthropic_api_key="test-key"))
    monkeypatch.setattr(LLMClient, "anthropic", _fake_anthropic([], captured))

    assert await client.complete("hi") == ""
    assert captured["system"] == llm.DEFAULT_SYSTEM_PROMPT


async def test_anthropic_property_injects_monkeypatched_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _create(**kwargs: Any) -> Any:
        return SimpleNamespace(content=[TextBlock(text="sdk text", type="text")])

    class _FakeSDK:
        def __init__(self, **kwargs: Any) -> None:
            captured["init"] = kwargs
            self.messages = SimpleNamespace(create=_create)

    monkeypatch.setattr(llm, "AsyncAnthropic", _FakeSDK)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")

    assert await LLMClient(Settings(anthropic_api_key=None)).complete("hi") == "sdk text"
    assert captured["init"] == {"api_key": "sk-from-env"}


# --------------------------------------------------------------------------- retries


@pytest.mark.parametrize(
    ("failures", "expected_attempts"),
    [(0, 1), (1, 2), (2, 3), (3, 4)],
)
async def test_transient_errors_are_retried(
    no_backoff: None, failures: int, expected_attempts: int
) -> None:
    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] <= failures:
            raise RuntimeError("rate limit exceeded")
        return "ok"

    client = LLMClient(Settings(llm_max_retries=3))
    assert await client._with_retries(flaky) == "ok"
    assert attempts["n"] == expected_attempts


async def test_retries_are_bounded_by_llm_max_retries(no_backoff: None) -> None:
    attempts = {"n": 0}

    async def always_timeout() -> str:
        attempts["n"] += 1
        raise httpx.ReadTimeout("timed out")

    client = LLMClient(Settings(llm_max_retries=2))
    with pytest.raises(httpx.ReadTimeout):
        await client._with_retries(always_timeout)
    assert attempts["n"] == 3  # llm_max_retries + 1


@pytest.mark.parametrize(
    "exc",
    [ValueError("ANTHROPIC_API_KEY is not set"), AuthError("invalid api key")],
    ids=["local-value-error", "auth-401"],
)
async def test_non_transient_errors_are_attempted_once(
    no_backoff: None, exc: Exception
) -> None:
    attempts = {"n": 0}

    async def failing() -> str:
        attempts["n"] += 1
        raise exc

    client = LLMClient(Settings(llm_max_retries=5))
    with pytest.raises(type(exc)):
        await client._with_retries(failing)
    assert attempts["n"] == 1


# --------------------------------------------------------------------------- ollama http


async def test_complete_ollama_posts_to_chat_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Response:
        def raise_for_status(self) -> None:
            captured["raised"] = True

        def json(self) -> dict[str, Any]:
            return {"message": {"content": "ollama says hi"}}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def post(self, url: str, json: Any) -> _Response:
            captured["url"] = url
            captured["json"] = json
            return _Response()

    monkeypatch.setattr(llm.httpx, "AsyncClient", _FakeClient)
    client = LLMClient(
        Settings(
            provider=Provider.OLLAMA,
            ollama_base_url="http://host:1234/",
            ollama_model="llama-x",
        )
    )

    assert await client.complete("hi", system="sys") == "ollama says hi"
    assert captured["url"] == "http://host:1234/api/chat"
    assert captured["timeout"] == llm.HTTP_TIMEOUT_SECONDS
    assert captured["raised"] is True
    assert captured["json"] == {
        "model": "llama-x",
        "stream": False,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
    }


# --------------------------------------------------------------------------- files


async def test_write_then_read_round_trip(tmp_path: Any) -> None:
    client = LLMClient()
    target = tmp_path / "nested" / "notes.txt"
    await client.write_file(target, "hello\nworld")
    assert target.read_text() == "hello\nworld"
    assert await client.read_file(target) == "hello\nworld"


async def test_read_file_missing_raises(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError, match="File not found"):
        await LLMClient().read_file(tmp_path / "absent.txt")


def test_list_files_relative_and_prunes_noise(tmp_path: Any) -> None:
    for relative in (
        "a.txt",
        "pkg/b.py",
        ".git/config",
        "__pycache__/a.cpython-312.pyc",
        "node_modules/dep/index.js",
        ".venv/bin/python",
        ".pytest_cache/v/cache/nodeids",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")

    assert LLMClient().list_files(tmp_path) == ["a.txt", "pkg/b.py"]
    assert LLMClient().list_files(tmp_path / "does-not-exist") == []
