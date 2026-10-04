"""Tests for :mod:`furrow.llm`.

The Anthropic/OpenAI SDKs are *optional* imports in ``furrow.llm``: the module
attributes ``AsyncAnthropic`` / ``AsyncOpenAI`` are ``None`` when a package is
missing.  These tests therefore monkeypatch those module attributes with
recording factories instead of importing the real SDKs, so the suite behaves
identically with and without them installed.  ``pytest.importorskip`` is
deliberately *not* used.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from furrow import llm as llm_module
from furrow.config import Provider, Settings
from furrow.llm import OLLAMA_PLACEHOLDER_API_KEY, LLMClient

_API_KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

# A hermetic baseline: no .env file, no inherited FURROW_* values.
_BASELINE = Settings(_env_file=None)


def make_settings(**overrides: Any) -> Settings:
    """Build a Settings object that ignores .env / the ambient environment."""
    values: dict[str, Any] = {
        "provider": Provider.ANTHROPIC,
        "model": "settings-model",
        "anthropic_api_key": "anthropic-secret",
        "openai_api_key": "openai-secret",
        "max_cycles": 0,
        "max_parallel_tasks": 2,
        "workspace": Path("/tmp"),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class Recorder:
    """Captures SDK constructor kwargs and every ``create()`` call."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.constructed: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def sdk_factory(self, sdk: str) -> type:
        """Return a drop-in replacement for the named SDK client class."""
        recorder = self

        class FakeSDK:
            def __init__(self, **kwargs: Any) -> None:
                recorder.constructed.append({"sdk": sdk, **kwargs})

            @property
            def messages(self) -> FakeSDK:
                return self

            @property
            def chat(self) -> FakeSDK:
                return self

            @property
            def completions(self) -> FakeSDK:
                return self

            async def create(self, **kwargs: Any) -> Any:
                recorder.calls.append({"sdk": sdk, **kwargs})
                if sdk == "anthropic":
                    return SimpleNamespace(content=[SimpleNamespace(text=recorder.reply)])
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content=recorder.reply))
                    ]
                )

        return FakeSDK


@pytest.fixture(autouse=True)
def _no_ambient_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The developer's real keys must not leak into the assertions."""
    for name in _API_KEY_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(llm_module, "AsyncAnthropic", rec.sdk_factory("anthropic"))
    monkeypatch.setattr(llm_module, "AsyncOpenAI", rec.sdk_factory("openai"))
    return rec


@pytest.mark.parametrize("provider", list(Provider), ids=[p.value for p in Provider])
async def test_complete_dispatches_for_every_provider(
    provider: Provider, recorder: Recorder
) -> None:
    """Dispatch is total: no Provider member falls through to a crash."""
    client = LLMClient(make_settings(provider=provider))
    recorder.reply = f"reply from {provider.value}"

    assert await client.complete("hello") == f"reply from {provider.value}"

    assert recorder.calls, f"{provider.value} never reached an SDK create() call"
    assert recorder.calls[-1]["sdk"] == (
        "anthropic" if provider is Provider.ANTHROPIC else "openai"
    )
    assert recorder.calls[-1]["model"] == "settings-model"


async def test_ollama_client_uses_base_url_and_placeholder_key(recorder: Recorder) -> None:
    settings = make_settings(provider=Provider.OLLAMA)
    client = LLMClient(settings)

    await client.complete("hello")

    assert recorder.constructed == [
        {
            "sdk": "openai",
            "base_url": f"{settings.ollama_base_url}/v1",
            "api_key": OLLAMA_PLACEHOLDER_API_KEY,
        }
    ]
    assert OLLAMA_PLACEHOLDER_API_KEY == "ollama"


async def test_ollama_base_url_is_read_from_settings(recorder: Recorder) -> None:
    client = LLMClient(
        make_settings(provider=Provider.OLLAMA, ollama_base_url="http://ollama.invalid:1234")
    )

    await client.complete("hello")

    assert recorder.constructed[-1]["base_url"] == "http://ollama.invalid:1234/v1"
    assert recorder.constructed[-1]["base_url"] != f"{_BASELINE.ollama_base_url}/v1"


async def test_ollama_completes_with_no_api_key_env_vars(recorder: Recorder) -> None:
    client = LLMClient(
        make_settings(provider=Provider.OLLAMA, anthropic_api_key=None, openai_api_key=None)
    )

    for name in _API_KEY_VARS:
        assert name not in os.environ

    assert await client.complete("hello") == "ok"
    assert recorder.calls[-1]["sdk"] == "openai"


@pytest.mark.parametrize(
    ("provider", "primary", "prefixed"),
    [
        (Provider.ANTHROPIC, "ANTHROPIC_API_KEY is not set", "FURROW_ANTHROPIC_API_KEY"),
        (Provider.OPENAI, "OPENAI_API_KEY is not set", "FURROW_OPENAI_API_KEY"),
    ],
    ids=["anthropic", "openai"],
)
async def test_missing_api_key_names_both_env_vars(
    provider: Provider, primary: str, prefixed: str, recorder: Recorder
) -> None:
    client = LLMClient(
        make_settings(provider=provider, anthropic_api_key=None, openai_api_key=None)
    )

    with pytest.raises(ValueError) as excinfo:
        await client.complete("hello")

    message = str(excinfo.value)
    # The first substring is contained in the second; assert both deliberately.
    assert primary in message
    assert prefixed in message
    assert not recorder.constructed


@pytest.mark.parametrize("provider", list(Provider), ids=[p.value for p in Provider])
async def test_model_falls_back_to_settings_model(provider: Provider, recorder: Recorder) -> None:
    client = LLMClient(make_settings(provider=provider, model="fallback-model"))

    await client.complete("hello")

    assert recorder.calls[-1]["model"] == "fallback-model"


async def test_explicit_model_override_is_honored(recorder: Recorder) -> None:
    client = LLMClient(make_settings(provider=Provider.ANTHROPIC, model="fallback-model"))

    await client.complete("hello", model="override-model")

    assert recorder.calls[-1]["model"] == "override-model"


async def test_unknown_provider_is_rejected_with_full_provider_list(recorder: Recorder) -> None:
    client = LLMClient(make_settings(provider=Provider.ANTHROPIC))
    # Bypass Settings validation to reach the dispatch fallback.
    client.settings.provider = "gemini"

    with pytest.raises(ValueError) as excinfo:
        await client.complete("hello")

    message = str(excinfo.value)
    assert "Unsupported provider: gemini" in message
    for supported in Provider:
        assert supported.value in message
    assert not recorder.constructed


async def test_read_file_round_trip(tmp_path: Path) -> None:
    client = LLMClient(make_settings())
    target = tmp_path / "nested" / "note.txt"

    await client.write_file(target, "hello file")

    assert await client.read_file(target) == "hello file"
    assert client.list_files(tmp_path) == ["nested/note.txt"]
    assert client.list_files(tmp_path / "missing") == []
