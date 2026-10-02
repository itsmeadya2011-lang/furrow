"""Offline tests for :mod:`furrow.config`, :mod:`furrow.util` and :mod:`furrow.llm`.

These tests must stay offline and deterministic: no network, no real API keys, no
real-time ``asyncio.sleep``, no reliance on a developer's ``.env`` file, and no
dependence on ``settings.workspace`` pointing at the repository.  Every provider
SDK object is replaced with a fake, and every filesystem fixture lives in
``tmp_path``.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, call

import anthropic
import httpx
import openai
import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, RunResult, Settings, TaskModel, TaskStatus, TestResult
from furrow.llm import LLMClient, LLMError
from furrow.util import (
    JSONExtractionError,
    PathEscapeError,
    extract_json,
    extract_json_object,
    resolve_within,
    truncate,
)

# --------------------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def no_ambient_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop provider keys from the environment so the key-checking paths are exercised."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A throwaway sandbox root; nothing outside of it is ever created."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def settings(workspace: Path) -> Settings:
    """Deterministic settings: temp workspace, no keys, ``.env`` file ignored."""
    return Settings(
        workspace=workspace,
        provider=Provider.ANTHROPIC,
        anthropic_api_key=None,
        openai_api_key=None,
        _env_file=None,
    )


@pytest.fixture
def make_client(workspace: Path):
    """Build a client with hermetic settings, optionally overriding any field."""

    def _make(**overrides: Any) -> LLMClient:
        return LLMClient(Settings(workspace=workspace, _env_file=None, **overrides))

    return _make


def fake_anthropic(*, text: str = "ok", blocks: list[Any] | None = None) -> Any:
    """A stand-in for ``AsyncAnthropic`` whose ``messages.create`` is an ``AsyncMock``."""
    if blocks is None:
        blocks = [SimpleNamespace(type="text", text=text)]
    response = SimpleNamespace(content=blocks)
    return SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=response)),
        close=AsyncMock(),
    )


def fake_openai(*, content: str | None = "ok", choices: list[Any] | None = None) -> Any:
    """A stand-in for ``AsyncOpenAI`` whose ``chat.completions.create`` is an ``AsyncMock``."""
    if choices is None:
        choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
    response = SimpleNamespace(choices=choices)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))),
        close=AsyncMock(),
    )


def install_client(monkeypatch: pytest.MonkeyPatch, name: str, client: Any) -> Any:
    """Replace a lazily built provider property on :class:`LLMClient` with a fixed fake."""
    monkeypatch.setattr(
        LLMClient,
        name,
        property(lambda _self, _fake=client: _fake),
        raising=False,
    )
    return client


def status_error(exc_type: type[BaseException], message: str, status_code: int) -> BaseException:
    """Build a real provider SDK error (no socket involved)."""
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://provider.invalid/v1/messages"),
        json={"error": message},
    )
    return exc_type(message, response=response, body=None)


# ------------------------------------------------------------- furrow.util: extract_json


def test_extract_json_parses_a_plain_json_object() -> None:
    assert extract_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_extract_json_parses_a_fenced_json_block_surrounded_by_prose() -> None:
    reply = 'Here is the plan:\n\n```json\n{"tasks": ["a", "b"]}\n```\n\nLet me know.'
    assert extract_json(reply) == {"tasks": ["a", "b"]}


def test_extract_json_finds_an_object_embedded_in_a_longer_message() -> None:
    reply = 'Sure! {"note": "a } brace lives in here", "ok": true} -- hope that helps.'
    assert extract_json(reply) == {"note": "a } brace lives in here", "ok": True}


def test_extract_json_handles_escaped_quotes_and_backslashes_inside_strings() -> None:
    raw = r'{"path": "C:\\tmp\\a", "quote": "he said \"hi\"", "brace": "}"}'
    assert extract_json(raw) == {"path": r"C:\tmp\a", "quote": 'he said "hi"', "brace": "}"}


def test_extract_json_parses_a_top_level_array() -> None:
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_json_raises_on_prose_that_contains_no_json() -> None:
    with pytest.raises(JSONExtractionError):
        extract_json("I am afraid I cannot help with that request.")


def test_extract_json_raises_on_an_empty_response() -> None:
    with pytest.raises(JSONExtractionError):
        extract_json("   \n  ")


def test_extract_json_rejects_a_non_string_input() -> None:
    with pytest.raises(JSONExtractionError):
        extract_json({"already": "parsed"})  # type: ignore[arg-type]


# -------------------------------------------------------- furrow.util: extract_json_object


def test_extract_json_object_returns_the_mapping() -> None:
    assert extract_json_object("prose {\"id\": 7} trailing") == {"id": 7}


def test_extract_json_object_unwraps_a_single_element_array_of_one_object() -> None:
    assert extract_json_object('[{"id": 7}]') == {"id": 7}


def test_extract_json_object_rejects_a_multi_element_array() -> None:
    with pytest.raises(JSONExtractionError):
        extract_json_object('[{"id": 7}, {"id": 8}]')


def test_extract_json_object_rejects_a_scalar() -> None:
    with pytest.raises(JSONExtractionError):
        extract_json_object("42")


# --------------------------------------------------------------------- furrow.util: truncate


def test_truncate_returns_short_text_unchanged() -> None:
    assert truncate("short", limit=100) == "short"


def test_truncate_returns_text_of_exactly_the_limit_unchanged() -> None:
    assert truncate("abcde", limit=5) == "abcde"


def test_truncate_over_the_limit_never_exceeds_the_limit() -> None:
    assert len(truncate("x" * 500, limit=120)) <= 120


def test_truncate_over_the_limit_keeps_both_head_and_tail() -> None:
    text = "HEAD-MARKER" + "y" * 400 + "TAIL-MARKER"
    result = truncate(text, limit=200)
    assert "HEAD-MARKER" in result
    assert "TAIL-MARKER" in result
    assert "[truncated]" in result


def test_truncate_with_a_non_positive_limit_returns_empty_string() -> None:
    assert truncate("anything", limit=0) == ""
    assert truncate("anything", limit=-10) == ""


# ------------------------------------------------------------------ furrow.util: paths


def test_resolve_within_resolves_a_relative_path_under_the_root(workspace: Path) -> None:
    resolved = resolve_within(workspace, "src/main.py")
    assert resolved == (workspace / "src" / "main.py").resolve()


def test_resolve_within_accepts_an_absolute_path_inside_the_root(workspace: Path) -> None:
    target = workspace / "src" / "main.py"
    assert resolve_within(workspace, target) == target.resolve()


def test_resolve_within_rejects_parent_directory_traversal(workspace: Path) -> None:
    with pytest.raises(PathEscapeError):
        resolve_within(workspace, "../escape.txt")


def test_resolve_within_rejects_an_absolute_path_outside_the_root(workspace: Path) -> None:
    with pytest.raises(PathEscapeError):
        resolve_within(workspace, "/etc/hostname")


def test_resolve_within_rejects_a_symlink_that_points_outside_the_root(workspace: Path) -> None:
    os.symlink("/etc/hostname", workspace / "link.txt")
    with pytest.raises(PathEscapeError):
        resolve_within(workspace, "link.txt")


# ------------------------------------------------------------------------ furrow.config


def test_task_status_completed_compares_equal_to_raw_string() -> None:
    assert TaskStatus.COMPLETED == "completed"


def test_task_status_skipped_compares_equal_to_raw_string() -> None:
    # The orchestrator treats SKIPPED like COMPLETED when counting finished work.
    assert TaskStatus.SKIPPED == "skipped"


def test_task_status_is_a_str_subclass_so_it_serialises_as_its_value() -> None:
    assert isinstance(TaskStatus.FAILED, str)
    assert json.dumps({"status": TaskStatus.FAILED}) == '{"status": "failed"}'


def test_task_model_defaults_to_pending_with_no_result() -> None:
    task = TaskModel(id="t1", description="write docs")
    assert task.files == []
    assert task.dependencies == []
    assert task.status is TaskStatus.PENDING
    assert task.result is None
    assert task.attempts == 0
    assert task.error is None


def test_task_model_default_lists_are_not_shared_between_instances() -> None:
    first = TaskModel(id="t1", description="a")
    first.files.append("a.py")
    assert TaskModel(id="t2", description="b").files == []


def test_plan_carries_its_tasks_and_rationale() -> None:
    plan = Plan(tasks=[TaskModel(id="t1", description="a")], rationale="because")
    assert [task.id for task in plan.tasks] == ["t1"]
    assert plan.rationale == "because"


def test_test_result_defaults_collect_no_failures_or_exit_code() -> None:
    result = TestResult(passed=True, summary="all good")
    assert result.failures == []
    assert result.exit_code is None
    assert result.command is None


def test_run_result_defaults_report_an_incomplete_run() -> None:
    result = RunResult(goal="ship it")
    assert result.cycles == 0
    assert result.completed is False
    assert result.reason == ""
    assert result.tests_passed is False
    assert result.tasks_completed == 0
    assert result.tasks_failed == 0


def test_settings_reads_the_furrow_prefixed_max_parallel_tasks_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    assert Settings(_env_file=None).max_parallel_tasks == 9


def test_settings_max_cycles_defaults_to_zero_which_means_unlimited() -> None:
    assert Settings(_env_file=None).max_cycles == 0


def test_settings_coerces_a_provider_string_into_the_enum() -> None:
    assert Settings(provider="ollama", _env_file=None).provider is Provider.OLLAMA


def test_settings_rejects_an_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        Settings(provider="not-a-provider", _env_file=None)


# ------------------------------------------------------------------- furrow.llm: clients


def test_anthropic_property_raises_when_no_api_key_is_available(make_client) -> None:
    client = make_client()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        _ = client.anthropic


def test_openai_property_raises_when_no_api_key_is_available(make_client) -> None:
    client = make_client()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        _ = client.openai


def test_anthropic_property_builds_and_caches_a_client_when_a_key_is_present(make_client) -> None:
    client = make_client(anthropic_api_key="test-key")
    first = client.anthropic
    assert first is client.anthropic


def test_ollama_property_builds_a_client_pointed_at_the_configured_base_url(
    make_client,
) -> None:
    client = make_client(ollama_base_url="http://127.0.0.1:12345")
    assert str(client.ollama.base_url) == "http://127.0.0.1:12345"


def test_default_model_uses_the_ollama_model_for_the_ollama_provider(make_client) -> None:
    client = make_client(provider=Provider.OLLAMA, ollama_model="qwen2.5-coder:7b")
    assert client.default_model() == "qwen2.5-coder:7b"


def test_default_model_uses_the_general_model_for_other_providers(make_client) -> None:
    client = make_client(provider=Provider.ANTHROPIC, model="claude-x")
    assert client.default_model() == "claude-x"


# -------------------------------------------------------------- furrow.llm: complete()


async def test_complete_sends_the_system_prompt_and_settings_max_tokens_to_anthropic(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_client(monkeypatch, "anthropic", fake_anthropic(text="hi"))
    out = await LLMClient(settings).complete("prompt", system="be terse")
    kwargs = fake.messages.create.await_args.kwargs
    assert out == "hi"
    assert kwargs["system"] == "be terse"
    assert kwargs["max_tokens"] == settings.max_tokens
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]


async def test_complete_substitutes_a_default_system_prompt_when_none_is_given(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_client(monkeypatch, "anthropic", fake_anthropic())
    await LLMClient(settings).complete("prompt")
    assert fake.messages.create.await_args.kwargs["system"] == "You are a helpful coding assistant."


async def test_complete_sends_a_system_message_before_the_user_message_for_openai(
    make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_client(monkeypatch, "openai", fake_openai(content="hi"))
    client = make_client(provider=Provider.OPENAI, openai_api_key="test-key")
    out = await client.complete("prompt", system="be terse")
    messages = fake.chat.completions.create.await_args.kwargs["messages"]
    assert out == "hi"
    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1] == {"role": "user", "content": "prompt"}


async def test_complete_prefers_explicit_model_max_tokens_and_temperature_over_settings(
    make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_client(monkeypatch, "openai", fake_openai())
    client = make_client(provider=Provider.OPENAI, model="settings-model", temperature=0.2)
    await client.complete("prompt", model="override", max_tokens=7, temperature=0.9)
    kwargs = fake.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "override"
    assert kwargs["max_tokens"] == 7
    assert kwargs["temperature"] == 0.9


async def test_complete_routes_the_ollama_provider_through_the_openai_compatible_client(
    make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    ollama_fake = install_client(monkeypatch, "ollama", fake_openai(content="local"))
    openai_fake = install_client(monkeypatch, "openai", fake_openai())
    client = make_client(provider=Provider.OLLAMA, ollama_model="qwen2.5-coder:7b")
    out = await client.complete("prompt")
    assert out == "local"
    assert ollama_fake.chat.completions.create.await_args.kwargs["model"] == "qwen2.5-coder:7b"
    assert openai_fake.chat.completions.create.await_count == 0


# ------------------------------------------------------- furrow.llm: response edge cases


async def test_complete_raises_when_anthropic_returns_no_content_blocks(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_client(monkeypatch, "anthropic", fake_anthropic(blocks=[]))
    with pytest.raises(LLMError, match="no text content"):
        await LLMClient(settings).complete("prompt")


async def test_complete_ignores_anthropic_non_text_blocks(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = [SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text="only this")]
    install_client(monkeypatch, "anthropic", fake_anthropic(blocks=blocks))
    assert await LLMClient(settings).complete("prompt") == "only this"


async def test_complete_raises_when_openai_returns_no_choices(
    make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_client(monkeypatch, "openai", fake_openai(choices=[]))
    client = make_client(provider=Provider.OPENAI, openai_api_key="test-key")
    with pytest.raises(LLMError, match="no choices"):
        await client.complete("prompt")


async def test_complete_returns_empty_string_when_openai_content_is_none(
    make_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_client(monkeypatch, "openai", fake_openai(content=None))
    client = make_client(provider=Provider.OPENAI, openai_api_key="test-key")
    assert await client.complete("prompt") == ""


async def test_complete_json_parses_the_model_reply(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install_client(monkeypatch, "anthropic", fake_anthropic(text='{"id": "t1"}'))
    assert await LLMClient(settings).complete_json("prompt") == {"id": "t1"}
    assert fake.messages.create.await_count == 1


async def test_complete_json_reports_unparseable_output_as_an_llm_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_client(monkeypatch, "anthropic", fake_anthropic(text="I refuse to answer."))
    with pytest.raises(LLMError, match="could not parse JSON"):
        await LLMClient(settings).complete_json("prompt")


# ----------------------------------------------------------------- furrow.llm: retries


@pytest.fixture
def instant_backoff(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Make retry sleeps instantaneous and their backoff deterministic."""
    monkeypatch.setattr(random, "uniform", lambda _low, _high: 0.0)
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    return sleep


async def test_complete_retries_rate_limits_and_grows_the_backoff(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, instant_backoff: AsyncMock
) -> None:
    response = SimpleNamespace(content=[SimpleNamespace(text="finally")])
    create = AsyncMock(
        side_effect=[
            status_error(anthropic.RateLimitError, "slow down", 429),
            status_error(anthropic.RateLimitError, "slow down", 429),
            response,
        ]
    )
    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    install_client(monkeypatch, "anthropic", fake)
    out = await LLMClient(settings).complete("prompt")
    assert out == "finally"
    assert create.await_count == 3
    # base 0.5s doubling: 0.5 then 1.0 (the jitter term is pinned to zero).
    assert instant_backoff.await_args_list == [call(0.5), call(1.0)]


async def test_complete_raises_llm_error_when_retries_are_exhausted(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, instant_backoff: AsyncMock
) -> None:
    create = AsyncMock(side_effect=status_error(anthropic.RateLimitError, "slow down", 429))
    install_client(monkeypatch, "anthropic", SimpleNamespace(messages=SimpleNamespace(create=create)))
    client = LLMClient(settings)
    with pytest.raises(LLMError) as excinfo:
        await client.complete("prompt")
    assert "anthropic" in str(excinfo.value)
    assert "3 attempts" in str(excinfo.value)
    assert create.await_count == client.settings.max_retries


async def test_complete_does_not_retry_an_authentication_error(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, instant_backoff: AsyncMock
) -> None:
    create = AsyncMock(
        side_effect=status_error(anthropic.AuthenticationError, "bad key", 401),
    )
    install_client(monkeypatch, "anthropic", SimpleNamespace(messages=SimpleNamespace(create=create)))
    with pytest.raises(LLMError, match="anthropic"):
        await LLMClient(settings).complete("prompt")
    assert create.await_count == 1
    instant_backoff.assert_not_awaited()


async def test_complete_does_not_retry_an_openai_authentication_error(
    make_client, monkeypatch: pytest.MonkeyPatch, instant_backoff: AsyncMock
) -> None:
    create = AsyncMock(
        side_effect=status_error(openai.AuthenticationError, "bad key", 401),
    )
    install_client(monkeypatch, "openai", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ))
    client = make_client(provider=Provider.OPENAI, openai_api_key="test-key")
    with pytest.raises(LLMError, match="openai"):
        await client.complete("prompt")
    assert create.await_count == 1
    instant_backoff.assert_not_awaited()


# ------------------------------------------------------------ furrow.llm: file access


async def test_read_file_truncates_to_max_bytes_keeping_both_ends(workspace: Path) -> None:
    target = workspace / "big.txt"
    target.write_text("HEAD" + "m" * 500 + "TAIL", encoding="utf-8")
    client = LLMClient(Settings(workspace=workspace, _env_file=None))
    content = await client.read_file(target, max_bytes=60)
    assert len(content) <= 60
    assert content.startswith("HEAD")
    assert content.endswith("TAIL")
    assert "[truncated]" in content


async def test_read_file_returns_the_whole_file_without_max_bytes(workspace: Path) -> None:
    target = workspace / "small.txt"
    target.write_text("exactly this", encoding="utf-8")
    client = LLMClient(Settings(workspace=workspace, _env_file=None))
    assert await client.read_file(target) == "exactly this"


async def test_write_file_creates_missing_parent_directories(workspace: Path) -> None:
    target = workspace / "deep" / "nested" / "out.txt"
    client = LLMClient(Settings(workspace=workspace, _env_file=None))
    await client.write_file(target, "written")
    assert target.parent.is_dir()
    assert target.read_text(encoding="utf-8") == "written"


async def test_write_file_overwrites_existing_content(workspace: Path) -> None:
    target = workspace / "out.txt"
    target.write_text("old", encoding="utf-8")
    client = LLMClient(Settings(workspace=workspace, _env_file=None))
    await client.write_file(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


# ------------------------------------------------------------ furrow.llm: list_files()


@pytest.fixture
def listings_client(workspace: Path) -> LLMClient:
    return LLMClient(Settings(workspace=workspace, _env_file=None))


def test_list_files_defaults_to_the_configured_workspace(
    workspace: Path, listings_client: LLMClient
) -> None:
    assert listings_client.list_files() == listings_client.list_files(workspace)
    assert "src/main.py" in listings_client.list_files()


def test_list_files_honours_ignore_dirs_and_ignore_globs(
    workspace: Path, listings_client: LLMClient
) -> None:
    for ignored_dir in (".git", "node_modules", "__pycache__"):
        (workspace / ignored_dir).mkdir()
        (workspace / ignored_dir / "junk.txt").write_text("x", encoding="utf-8")
    (workspace / "cached.pyc").write_bytes(b"\x00\x01")
    listed = listings_client.list_files()
    assert listed == ["src/main.py"]
    assert not any(".git" in name or "node_modules" in name for name in listed)
    assert not any(name.endswith(".pyc") for name in listed)


def test_list_files_skips_binary_files(workspace: Path, listings_client: LLMClient) -> None:
    (workspace / "blob.bin").write_bytes(b"\x89PNG\x00\x00\x00")
    assert listings_client.list_files() == ["src/main.py"]


def test_list_files_excludes_a_file_that_resolves_outside_the_root(
    workspace: Path, listings_client: LLMClient
) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    os.symlink(outside, workspace / "escape.txt")
    try:
        assert listings_client.list_files() == ["src/main.py"]
    finally:
        outside.unlink()


def test_list_files_returns_sorted_relative_paths(
    workspace: Path, listings_client: LLMClient
) -> None:
    (workspace / "b.txt").write_text("b", encoding="utf-8")
    (workspace / "a.txt").write_text("a", encoding="utf-8")
    listed = listings_client.list_files()
    assert listed == sorted(listed)
    assert listed == ["a.txt", "b.txt", "src/main.py"]


def test_list_files_respects_max_files(workspace: Path, listings_client: LLMClient) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        (workspace / name).write_text(name, encoding="utf-8")
    assert len(listings_client.list_files(max_files=2)) == 2


def test_list_files_can_include_directory_names(
    workspace: Path, listings_client: LLMClient
) -> None:
    listed = listings_client.list_files(include_dirs=True)
    assert "src" in listed


def test_list_files_returns_empty_for_a_directory_that_does_not_exist(
    workspace: Path, listings_client: LLMClient
) -> None:
    assert listings_client.list_files(workspace / "nope") == []


async def test_aclose_closes_and_forgets_cached_clients(workspace: Path) -> None:
    client = LLMClient(Settings(workspace=workspace, _env_file=None))
    fake = fake_anthropic()
    client._anthropic = fake
    await client.aclose()
    fake.close.assert_awaited_once()
    assert client._anthropic is None