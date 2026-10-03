"""The LLM client: settings fallback, provider routing, retries, file helpers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from furrow import config
from furrow import llm as llm_module
from furrow.llm import LLMClient
from tests.conftest import FakeLLM, make_settings


class RateLimitError(Exception):
    """Stands in for a provider SDK rate-limit error."""


class FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def install_fake_anthropic(monkeypatch: pytest.MonkeyPatch, text: str = "anthropic reply"):
    """Replace the Anthropic SDK class; returns the recorded constructor kwargs."""
    created: list[dict[str, Any]] = []

    def factory(**kwargs: Any):
        created.append(kwargs)

        async def create(**call: Any):
            created.append(call)
            return SimpleNamespace(content=[SimpleNamespace(text=text)])

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(llm_module, "AsyncAnthropic", factory)
    return created


def install_fake_openai(monkeypatch: pytest.MonkeyPatch, text: str = "openai reply"):
    """Replace the OpenAI SDK class; returns the recorded constructor kwargs."""
    created: list[dict[str, Any]] = []

    def factory(**kwargs: Any):
        created.append(kwargs)

        async def create(**call: Any):
            created.append(call)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr(llm_module, "AsyncOpenAI", factory)
    return created


def count_attempts(monkeypatch: pytest.MonkeyPatch, exc_factory):
    """Make every completion attempt raise; returns the list that records attempts."""
    attempts: list[tuple[str, str | None]] = []

    async def _complete_once(self, prompt: str, system: str, model: str):
        attempts.append((prompt, model))
        raise exc_factory()

    monkeypatch.setattr(llm_module.LLMClient, "_complete_once", _complete_once)
    return attempts


# --------------------------------------------------------------------- settings


def test_default_client_falls_back_to_the_module_singleton() -> None:
    # Regression: ``__init__`` used to store ``None``, so every attribute access
    # raised ``AttributeError: 'NoneType' object has no attribute ...``.
    client = LLMClient()
    assert client.settings is config.settings
    assert isinstance(client.settings, config.Settings)


def test_explicit_none_falls_back_to_the_module_singleton() -> None:
    client = LLMClient(settings=None)
    assert client.settings is config.settings
    assert client.settings.model  # would raise AttributeError before the fix


def test_explicit_settings_are_used_verbatim() -> None:
    settings = make_settings(model="custom-model")
    client = LLMClient(settings=settings)
    assert client.settings is settings
    assert client.settings.model == "custom-model"


def test_agents_constructible_without_a_client() -> None:
    # The user-visible symptom of the None-settings bug: constructing any agent
    # with its own default client used to raise AttributeError on first use.
    from furrow.agents.planner import PlannerAgent
    from furrow.agents.tester import TesterAgent
    from furrow.agents.worker import WorkerAgent

    task = config.TaskModel(id="1", description="do it")
    for agent in (PlannerAgent(), WorkerAgent(task=task), TesterAgent()):
        assert isinstance(agent.client.settings, config.Settings)


def test_the_fake_client_fixture_is_a_scripted_llm_client(fake_client) -> None:
    assert isinstance(fake_client, LLMClient)
    assert isinstance(fake_client.settings, config.Settings)
    # The fixture is a FakeLLM: it records every call it is handed.
    assert isinstance(fake_client, FakeLLM)
    assert fake_client.calls == []  # nothing has been asked of it yet


# ------------------------------------------------------------------- providers


def test_anthropic_property_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_anthropic(monkeypatch)
    client = LLMClient(settings=make_settings(anthropic_api_key=None))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


def test_openai_property_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_openai(monkeypatch)
    client = LLMClient(settings=make_settings(openai_api_key=None))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


def test_anthropic_client_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_fake_anthropic(monkeypatch)
    client = LLMClient(settings=make_settings(anthropic_api_key="sk-test"))
    first = client.anthropic
    assert client.anthropic is first
    assert len(created) == 1  # the SDK class was constructed exactly once
    assert created[0] == {"api_key": "sk-test"}


def test_openai_client_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_fake_openai(monkeypatch)
    client = LLMClient(settings=make_settings(openai_api_key="sk-test"))
    first = client.openai
    assert client.openai is first
    assert len(created) == 1
    assert created[0] == {"api_key": "sk-test"}


def test_anthropic_completion_returns_the_first_text_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = install_fake_anthropic(monkeypatch, text="hello from claude")
    client = LLMClient(settings=make_settings(anthropic_api_key="sk-test", model="claude-x"))
    result = asyncio.run(client.complete("hi", system="be terse"))
    assert result == "hello from claude"
    call = created[-1]
    assert call["model"] == "claude-x"
    assert call["system"] == "be terse"
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_completion_returns_the_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_fake_openai(monkeypatch, text="hello from gpt")
    client = LLMClient(
        settings=make_settings(provider=config.Provider.OPENAI, openai_api_key="sk-test")
    )
    result = asyncio.run(client.complete("hi", model="gpt-x"))
    assert result == "hello from gpt"
    call = created[-1]
    assert call["model"] == "gpt-x"
    assert call["messages"][0] == {
        "role": "system",
        "content": "You are a helpful coding assistant.",
    }


def test_ollama_routes_through_the_openai_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: ``Provider.OLLAMA`` used to raise "Unsupported provider".
    created = install_fake_openai(monkeypatch, text="hello from llama")
    settings = make_settings(
        provider=config.Provider.OLLAMA,
        model="llama3.2",
        ollama_base_url="http://ollama.invalid:1234",
    )
    result = asyncio.run(LLMClient(settings=settings).complete("hi"))
    assert result == "hello from llama"
    assert created[0]["base_url"] == "http://ollama.invalid:1234"
    assert created[0]["api_key"] == "ollama"  # Ollama ignores the key but OpenAI demands one
    assert created[-1]["model"] == "llama3.2"


def test_ollama_client_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    created = install_fake_openai(monkeypatch)
    client = LLMClient(settings=make_settings(provider=config.Provider.OLLAMA))
    assert client.ollama is client.ollama
    assert len(created) == 1


def test_unsupported_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings()
    # Bypass validation to reach the fallback branch of the provider match.
    settings.provider = "not-a-provider"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(LLMClient(settings=settings).complete("hi"))


# ---------------------------------------------------------------------- retries


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ValueError("no key"), False),
        (RateLimitError("slow down"), True),
        (FakeStatusError(429), True),
        (FakeStatusError(408), True),
        (FakeStatusError(500), True),
        (FakeStatusError(503), True),
        (FakeStatusError(400), False),
        (FakeStatusError(401), False),
        (RuntimeError("something else"), False),
    ],
)
def test_transient_predicate(exc: BaseException, expected: bool) -> None:
    assert llm_module._is_transient(exc) is expected


def test_transient_errors_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = count_attempts(monkeypatch, RateLimitError)
    client = LLMClient(settings=make_settings(max_retries=1))
    with pytest.raises(RateLimitError):
        asyncio.run(client.complete("hi"))
    assert len(attempts) == 2  # the initial try plus one retry


def test_value_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = count_attempts(monkeypatch, lambda: ValueError("ANTHROPIC_API_KEY is not set"))
    client = LLMClient(settings=make_settings(max_retries=3))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        asyncio.run(client.complete("hi"))
    assert len(attempts) == 1  # fail fast, no pointless retries


def test_unknown_provider_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = count_attempts(monkeypatch, lambda: ValueError("Unsupported provider: 'nope'"))
    settings = make_settings(max_retries=2)
    settings.provider = "nope"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(LLMClient(settings=settings).complete("hi"))
    assert len(attempts) == 1


def test_max_retries_controls_the_attempt_count(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = count_attempts(monkeypatch, RateLimitError)
    client = LLMClient(settings=make_settings(max_retries=0))
    with pytest.raises(RateLimitError):
        asyncio.run(client.complete("hi"))
    assert len(attempts) == 1


def test_successful_completion_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    async def _complete_once(self, prompt, system, model):
        calls.append(1)
        return "fine"

    monkeypatch.setattr(llm_module.LLMClient, "_complete_once", _complete_once)
    client = LLMClient(settings=make_settings(max_retries=5))
    assert asyncio.run(client.complete("hi")) == "fine"
    assert len(calls) == 1


# ----------------------------------------------------------------- file helpers


def test_list_files_skips_noise_directories(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x", encoding="utf-8")
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    for junk in ("__pycache__", ".git", ".venv", "venv", "node_modules", ".hidden"):
        directory = tmp_path / junk
        directory.mkdir()
        (directory / "junk.py").write_text("x", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")

    assert LLMClient().list_files(tmp_path) == ["a.py", "src/app.py"]


def test_list_files_returns_sorted_posix_paths(tmp_path: Path) -> None:
    for name in ("z.txt", "a.txt", "b/c.txt", "b/a.txt"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")

    assert LLMClient().list_files(tmp_path) == ["a.txt", "b/a.txt", "b/c.txt", "z.txt"]


def test_list_files_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert LLMClient().list_files(tmp_path / "nope") == []


def test_list_files_of_an_empty_directory_is_empty(tmp_path: Path) -> None:
    assert LLMClient().list_files(tmp_path) == []


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    client = LLMClient()
    target = tmp_path / "pkg" / "mod.py"
    asyncio.run(client.write_file(target, "value = 1\n# café\n"))
    assert asyncio.run(client.read_file(target)) == "value = 1\n# café\n"


def test_write_file_overwrites_and_creates_parents(tmp_path: Path) -> None:
    client = LLMClient()
    target = tmp_path / "deep" / "nested" / "dir" / "out.txt"
    asyncio.run(client.write_file(target, "first"))
    asyncio.run(client.write_file(target, "second"))
    assert asyncio.run(client.read_file(target)) == "second"
    assert target.parent.is_dir()


def test_read_file_accepts_strings(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    assert asyncio.run(LLMClient().read_file(str(target))) == "hello"


def test_read_file_of_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        asyncio.run(LLMClient().read_file(tmp_path / "missing.txt"))
