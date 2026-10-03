"""Tests for :mod:`furrow.llm` -- provider dispatch, retries and file helpers.

Everything here is offline. The provider SDKs are never asked to talk to
anything: the lazy ``anthropic`` / ``openai`` / ``ollama`` properties are fed
stand-in client objects, and the autouse ``no_network_egress`` guard in
``conftest.py`` turns any accidental socket into a loud failure. Retry tests
keep tenacity's backoff disabled (via :func:`tests.conftest.no_retry_sleep`) so
they assert the attempt budget, not the wall clock.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import openai
import pytest
from anthropic.types import TextBlock, ThinkingBlock

from furrow.config import Provider
from furrow.llm import (
    DEFAULT_IGNORE,
    MAX_TOKENS_DEFAULT,
    RETRY_ATTEMPTS,
    LLMClient,
    anthropic_text,
    is_retryable_error,
)
from tests.conftest import (
    SettingsFactory,
    no_retry_sleep,
    stub_request,
    stub_response,
)

# --------------------------------------------------------------------------- #
# Stand-in SDK clients
# --------------------------------------------------------------------------- #


class FakeAnthropicSDK:
    """Just enough of ``AsyncAnthropic`` for :meth:`LLMClient.complete`."""

    def __init__(
        self,
        replies: list[str] | None = None,
        errors: list[BaseException] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self.create)
        self._replies = list(replies or [])
        self._errors = list(errors or [])

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        text = self._replies.pop(0) if self._replies else "ok"
        return SimpleNamespace(content=[TextBlock(text=text, type="text")])


class FakeOpenAISDK:
    """Just enough of ``AsyncOpenAI`` for :meth:`LLMClient.complete`."""

    def __init__(self, reply: str = "ok", empty: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self._reply = reply
        self._empty = empty

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._empty:
            return SimpleNamespace(choices=[])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._reply))]
        )


def install_sdks(client: LLMClient, **sdks: Any) -> None:
    """Fill the client's lazy provider slots with stand-in SDK objects."""
    for name, sdk in sdks.items():
        setattr(client, f"_{name}", sdk)


def anthropic_error(
    error_class: type[anthropic.APIStatusError],
    status_code: int,
) -> anthropic.APIStatusError:
    # The SDK type hints point at httpx2; the response is only read, never sent.
    return error_class(
        "stub failure",
        response=stub_response(status_code),  # type: ignore[arg-type]
        body=None,
    )


def openai_error(
    error_class: type[openai.APIStatusError],
    status_code: int,
) -> openai.APIStatusError:
    return error_class(
        "stub failure",
        response=stub_response(status_code),  # type: ignore[arg-type]
        body=None,
    )


# --------------------------------------------------------------------------- #
# Provider dispatch
# --------------------------------------------------------------------------- #


async def test_complete_uses_the_anthropic_sdk_for_the_anthropic_provider(
    settings_factory: SettingsFactory,
) -> None:
    client = LLMClient(settings=settings_factory(provider=Provider.ANTHROPIC))
    anthropic_sdk = FakeAnthropicSDK(replies=["from anthropic"])
    openai_sdk = FakeOpenAISDK(reply="from openai")
    ollama_sdk = FakeOpenAISDK(reply="from ollama")
    install_sdks(client, anthropic=anthropic_sdk, openai=openai_sdk, ollama=ollama_sdk)

    reply = await client.complete("hello")

    assert reply == "from anthropic"
    assert len(anthropic_sdk.calls) == 1
    assert openai_sdk.calls == []
    assert ollama_sdk.calls == []


async def test_complete_uses_the_openai_sdk_for_the_openai_provider(
    settings_factory: SettingsFactory,
) -> None:
    client = LLMClient(settings=settings_factory(provider=Provider.OPENAI))
    anthropic_sdk = FakeAnthropicSDK(replies=["from anthropic"])
    openai_sdk = FakeOpenAISDK(reply="from openai")
    ollama_sdk = FakeOpenAISDK(reply="from ollama")
    install_sdks(client, anthropic=anthropic_sdk, openai=openai_sdk, ollama=ollama_sdk)

    reply = await client.complete("hello")

    assert reply == "from openai"
    assert len(openai_sdk.calls) == 1
    assert anthropic_sdk.calls == []
    assert ollama_sdk.calls == []


async def test_complete_uses_the_ollama_sdk_for_the_ollama_provider(
    settings_factory: SettingsFactory,
) -> None:
    client = LLMClient(settings=settings_factory(provider=Provider.OLLAMA))
    anthropic_sdk = FakeAnthropicSDK(replies=["from anthropic"])
    openai_sdk = FakeOpenAISDK(reply="from openai")
    ollama_sdk = FakeOpenAISDK(reply="from ollama")
    install_sdks(client, anthropic=anthropic_sdk, openai=openai_sdk, ollama=ollama_sdk)

    reply = await client.complete("hello")

    assert reply == "from ollama"
    assert len(ollama_sdk.calls) == 1
    assert anthropic_sdk.calls == []
    assert openai_sdk.calls == []


async def test_complete_defaults_to_the_configured_model(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(model="configured-model")
    client = LLMClient(settings=settings)
    anthropic_sdk = FakeAnthropicSDK()
    install_sdks(client, anthropic=anthropic_sdk)

    await client.complete("hello")

    assert anthropic_sdk.calls[0]["model"] == "configured-model"


async def test_complete_prefers_an_explicit_model_argument(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(model="configured-model")
    client = LLMClient(settings=settings)
    anthropic_sdk = FakeAnthropicSDK()
    install_sdks(client, anthropic=anthropic_sdk)

    await client.complete("hello", model="per-call-model")

    assert anthropic_sdk.calls[0]["model"] == "per-call-model"


async def test_complete_forwards_the_configured_token_cap(
    settings_factory: SettingsFactory,
) -> None:
    settings = settings_factory(max_tokens=321)
    client = LLMClient(settings=settings)
    anthropic_sdk = FakeAnthropicSDK()
    install_sdks(client, anthropic=anthropic_sdk)

    await client.complete("hello")

    assert client.max_tokens == 321
    assert anthropic_sdk.calls[0]["max_tokens"] == 321


async def test_complete_sends_the_prompt_as_a_user_message(
    settings_factory: SettingsFactory,
) -> None:
    client = LLMClient(settings=settings_factory())
    anthropic_sdk = FakeAnthropicSDK()
    install_sdks(client, anthropic=anthropic_sdk)

    await client.complete("the prompt", system="the system")

    call = anthropic_sdk.calls[0]
    assert call["messages"] == [{"role": "user", "content": "the prompt"}]
    assert call["system"] == "the system"


async def test_complete_returns_an_empty_string_when_the_provider_replies_with_no_choices(
    settings_factory: SettingsFactory,
) -> None:
    client = LLMClient(settings=settings_factory(provider=Provider.OPENAI))
    install_sdks(client, openai=FakeOpenAISDK(empty=True))
    assert await client.complete("hello") == ""


async def test_complete_returns_an_empty_string_when_the_model_produces_no_text(
    settings_factory: SettingsFactory,
) -> None:
    client = LLMClient(settings=settings_factory(provider=Provider.ANTHROPIC))
    install_sdks(client, anthropic=FakeAnthropicSDK(replies=[""]))
    assert await client.complete("hello") == ""


# --------------------------------------------------------------------------- #
# Client construction (no request is issued)
# --------------------------------------------------------------------------- #


def test_ollama_client_targets_the_configured_base_url(
    settings_factory: SettingsFactory,
    clean_env: None,
) -> None:
    settings = settings_factory(
        provider=Provider.OLLAMA,
        ollama_base_url="http://127.0.0.1:11434",
        anthropic_api_key=None,
        openai_api_key=None,
    )
    ollama = LLMClient(settings=settings).ollama
    assert str(ollama.base_url).rstrip("/") == "http://127.0.0.1:11434"


def test_ollama_client_is_built_once(
    settings_factory: SettingsFactory,
    clean_env: None,
) -> None:
    client = LLMClient(settings=settings_factory(provider=Provider.OLLAMA))
    assert client.ollama is client.ollama


def test_anthropic_client_requires_an_api_key(
    settings_factory: SettingsFactory,
    clean_env: None,
) -> None:
    client = LLMClient(settings=settings_factory(anthropic_api_key=None))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY") as failure:
        _ = client.anthropic
    assert "anthropic" in str(failure.value)


def test_openai_client_requires_an_api_key(
    settings_factory: SettingsFactory,
    clean_env: None,
) -> None:
    client = LLMClient(settings=settings_factory(provider=Provider.OPENAI, openai_api_key=None))
    with pytest.raises(ValueError, match="OPENAI_API_KEY") as failure:
        _ = client.openai
    assert "openai" in str(failure.value)


# --------------------------------------------------------------------------- #
# anthropic_text
# --------------------------------------------------------------------------- #


def test_anthropic_text_returns_a_single_text_block() -> None:
    assert anthropic_text([TextBlock(text="hello", type="text")]) == "hello"


def test_anthropic_text_concatenates_several_text_blocks() -> None:
    content = [
        TextBlock(text="one ", type="text"),
        TextBlock(text="two ", type="text"),
        TextBlock(text="three", type="text"),
    ]
    assert anthropic_text(content) == "one two three"


def test_anthropic_text_skips_a_non_text_block_in_first_position() -> None:
    content = [
        ThinkingBlock(thinking="let me think", signature="sig", type="thinking"),
        TextBlock(text="the answer", type="text"),
    ]
    assert anthropic_text(content) == "the answer"


def test_anthropic_text_returns_an_empty_string_for_no_blocks() -> None:
    assert anthropic_text([]) == ""


def test_anthropic_text_returns_an_empty_string_when_nothing_is_text() -> None:
    content = [ThinkingBlock(thinking="hmm", signature="sig", type="thinking")]
    assert anthropic_text(content) == ""


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        pytest.param(anthropic_error(anthropic.RateLimitError, 429), True, id="anthropic-429"),
        pytest.param(anthropic_error(anthropic.APIStatusError, 503), True, id="anthropic-503"),
        pytest.param(anthropic_error(anthropic.APIStatusError, 400), False, id="anthropic-400"),
        pytest.param(anthropic_error(anthropic.APIStatusError, 401), False, id="anthropic-401"),
        pytest.param(openai_error(openai.RateLimitError, 429), True, id="openai-429"),
        pytest.param(openai_error(openai.APIStatusError, 500), True, id="openai-500"),
        pytest.param(openai_error(openai.APIStatusError, 404), False, id="openai-404"),
        pytest.param(
            anthropic.APIConnectionError(request=stub_request()),  # type: ignore[arg-type]
            True,
            id="anthropic-connection",
        ),
        pytest.param(
            openai.APIConnectionError(request=stub_request()),  # type: ignore[arg-type]
            True,
            id="openai-connection",
        ),
        pytest.param(RuntimeError("unrelated"), False, id="not-an-sdk-error"),
    ],
)
def test_is_retryable_error_separates_transient_failures_from_client_errors(
    error: BaseException,
    retryable: bool,
) -> None:
    assert is_retryable_error(error) is retryable


async def test_complete_succeeds_after_a_transient_failure(
    settings_factory: SettingsFactory,
) -> None:
    client = LLMClient(settings=settings_factory())
    anthropic_sdk = FakeAnthropicSDK(
        replies=["second attempt"],
        errors=[anthropic_error(anthropic.RateLimitError, 429)],
    )
    install_sdks(client, anthropic=anthropic_sdk)

    with no_retry_sleep():
        reply = await client.complete("hello")

    assert reply == "second attempt"
    assert len(anthropic_sdk.calls) == 2


async def test_complete_gives_up_after_the_bounded_attempt_budget(
    settings_factory: SettingsFactory,
) -> None:
    client = LLMClient(settings=settings_factory())
    anthropic_sdk = FakeAnthropicSDK(
        errors=[anthropic_error(anthropic.InternalServerError, 500)] * RETRY_ATTEMPTS
    )
    install_sdks(client, anthropic=anthropic_sdk)

    with pytest.raises(anthropic.InternalServerError), no_retry_sleep():
        await client.complete("hello")

    assert len(anthropic_sdk.calls) == RETRY_ATTEMPTS


async def test_complete_does_not_retry_a_client_error(
    settings_factory: SettingsFactory,
) -> None:
    client = LLMClient(settings=settings_factory())
    anthropic_sdk = FakeAnthropicSDK(
        errors=[anthropic_error(anthropic.BadRequestError, 400)] * RETRY_ATTEMPTS
    )
    install_sdks(client, anthropic=anthropic_sdk)

    with pytest.raises(anthropic.BadRequestError), no_retry_sleep():
        await client.complete("hello")

    assert len(anthropic_sdk.calls) == 1


def test_module_constants_match_the_documented_defaults() -> None:
    assert MAX_TOKENS_DEFAULT == 8192
    assert RETRY_ATTEMPTS == 3
    noise = {".git", "__pycache__", "*.pyc", ".venv", "node_modules"}
    assert noise <= DEFAULT_IGNORE


# --------------------------------------------------------------------------- #
# list_files
# --------------------------------------------------------------------------- #

KEPT_TREE = (
    "README.md",
    "notes.txt",
    "pkg/__init__.py",
    "pkg/module.py",
    "tests/test_pkg.py",
)
NOISE_TREE = (
    ".git/config",
    "__pycache__/module.cpython-310.pyc",
    ".venv/bin/python",
    "node_modules/left-pad/index.js",
    ".mypy_cache/3.1/cache.json",
    ".pytest_cache/v/cache/nodeids",
    ".ruff_cache/0.16/cache",
    ".furrow/state.json",
    "build/output.o",
    "pkg/module.pyc",
)


def build_tree(
    root: Path,
    entries: tuple[str, ...] = (*KEPT_TREE, *NOISE_TREE),
) -> Path:
    for entry in entries:
        target = root / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    return root


def test_list_files_skips_noise_directories(
    settings_factory: SettingsFactory,
    tmp_workspace: Path,
) -> None:
    build_tree(tmp_workspace)
    client = LLMClient(settings=settings_factory(workspace=tmp_workspace))
    assert client.list_files(tmp_workspace) == list(KEPT_TREE)


def test_list_files_honours_extra_ignore_patterns(
    settings_factory: SettingsFactory,
    tmp_workspace: Path,
) -> None:
    build_tree(tmp_workspace)
    client = LLMClient(settings=settings_factory(workspace=tmp_workspace))
    listed = client.list_files(tmp_workspace, ignore=["*.md", "tests"])
    assert listed == ["notes.txt", "pkg/__init__.py", "pkg/module.py"]


def test_list_files_returns_an_empty_list_for_a_missing_directory(
    settings_factory: SettingsFactory,
    tmp_path: Path,
) -> None:
    client = LLMClient(settings=settings_factory())
    assert client.list_files(tmp_path / "nope") == []


def test_list_files_accepts_a_string_path(
    settings_factory: SettingsFactory,
    tmp_workspace: Path,
) -> None:
    build_tree(tmp_workspace)
    client = LLMClient(settings=settings_factory(workspace=tmp_workspace))
    assert client.list_files(str(tmp_workspace)) == list(KEPT_TREE)


# --------------------------------------------------------------------------- #
# read_file / write_file
# --------------------------------------------------------------------------- #


async def test_write_file_then_read_file_round_trips(
    settings_factory: SettingsFactory,
    tmp_workspace: Path,
) -> None:
    client = LLMClient(settings=settings_factory(workspace=tmp_workspace))
    target = tmp_workspace / "notes.txt"
    await client.write_file(target, "hello\nworld\n")
    assert await client.read_file(target) == "hello\nworld\n"
    assert target.read_text(encoding="utf-8") == "hello\nworld\n"


async def test_write_file_creates_missing_parent_directories(
    settings_factory: SettingsFactory,
    tmp_workspace: Path,
) -> None:
    client = LLMClient(settings=settings_factory(workspace=tmp_workspace))
    target = tmp_workspace / "a" / "b" / "c.txt"
    await client.write_file(target, "deep")
    assert await client.read_file(target) == "deep"


async def test_write_file_leaves_no_temporary_file_behind(
    settings_factory: SettingsFactory,
    tmp_workspace: Path,
) -> None:
    client = LLMClient(settings=settings_factory(workspace=tmp_workspace))
    await client.write_file(tmp_workspace / "notes.txt", "content")
    assert sorted(entry.name for entry in tmp_workspace.iterdir()) == ["notes.txt"]


async def test_write_file_overwrites_existing_content(
    settings_factory: SettingsFactory,
    tmp_workspace: Path,
) -> None:
    client = LLMClient(settings=settings_factory(workspace=tmp_workspace))
    target = tmp_workspace / "notes.txt"
    await client.write_file(target, "first")
    await client.write_file(target, "second")
    assert await client.read_file(target) == "second"
    assert sorted(entry.name for entry in tmp_workspace.iterdir()) == ["notes.txt"]


async def test_read_file_accepts_a_string_path(
    settings_factory: SettingsFactory,
    tmp_workspace: Path,
) -> None:
    client = LLMClient(settings=settings_factory(workspace=tmp_workspace))
    target = tmp_workspace / "notes.txt"
    await client.write_file(target, "content")
    assert await client.read_file(str(target)) == "content"


async def test_read_file_reports_the_missing_path(
    settings_factory: SettingsFactory,
    tmp_workspace: Path,
) -> None:
    client = LLMClient(settings=settings_factory(workspace=tmp_workspace))
    missing = tmp_workspace / "absent.txt"
    with pytest.raises(FileNotFoundError) as failure:
        await client.read_file(missing)
    assert str(missing.resolve()) in str(failure.value)
