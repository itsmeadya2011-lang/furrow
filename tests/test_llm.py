"""Tests for the LLM client.

No network and no real provider client: dispatch is verified by swapping the
private provider coroutines for mocks, and each provider path is driven with an
injected fake -- a hand-built response object for Anthropic and OpenAI, and a
fake ``httpx.AsyncClient`` for the Ollama transport.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock

import pytest
from furrow.config import Provider, get_settings
from furrow.llm import _OLLAMA_TIMEOUT_SECONDS, LLMClient, _is_transient

# Directories the file listing must never descend into. ``build`` is here on
# purpose: it proves matching is per whole path component, so ``build.py`` (a
# file whose name merely starts with an ignored directory name) is kept.
_IGNORED_DIRS = [".git", "__pycache__", "node_modules", ".venv", "build"]


# --------------------------------------------------------------------------
# _is_transient
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_class",
    [ValueError, asyncio.CancelledError, KeyboardInterrupt, SystemExit],
)
def test_is_transient_is_false_for_permanent_errors(exc_class) -> None:
    assert _is_transient(exc_class()) is False


@pytest.mark.parametrize("exc_class", [RuntimeError, Exception])
def test_is_transient_is_true_for_provider_errors(exc_class) -> None:
    assert _is_transient(exc_class()) is True


@pytest.mark.parametrize("exc_class", [OSError, TimeoutError, asyncio.TimeoutError])
def test_is_transient_is_true_for_transport_errors(exc_class) -> None:
    # A timeout is worth retrying whichever class the runtime raises for it:
    # ``asyncio.TimeoutError`` is an alias of the builtin on 3.11+, while the
    # builtin is a subclass of ``OSError``.
    assert _is_transient(exc_class()) is True


# --------------------------------------------------------------------------
# list_files
# --------------------------------------------------------------------------


def _build_tree(root: Path) -> None:
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    (root / "build.py").write_text("build = 2\n", encoding="utf-8")
    (root / "pkg").mkdir()
    (root / "pkg" / "b.py").write_text("b = 1\n", encoding="utf-8")
    (root / "pkg" / "sub").mkdir()
    (root / "pkg" / "sub" / "c.py").write_text("c = 1\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep").mkdir()
    (root / "node_modules" / "dep" / "index.js").write_text("// js\n", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"\x00\x01")
    (root / ".venv").mkdir()
    (root / ".venv" / "lib").mkdir()
    (root / ".venv" / "lib" / "y.py").write_text("y = 1\n", encoding="utf-8")


def _client_for(root: Path) -> LLMClient:
    settings = get_settings().model_copy(
        update={"workspace": root, "ignored_dirs": list(_IGNORED_DIRS)}
    )
    return LLMClient(settings=settings)


def test_list_files_returns_relative_posix_paths(tmp_path) -> None:
    _build_tree(tmp_path)
    files = _client_for(tmp_path).list_files()
    assert "a.py" in files
    assert "pkg/b.py" in files
    assert "pkg/sub/c.py" in files


def test_list_files_skips_ignored_directories(tmp_path) -> None:
    _build_tree(tmp_path)
    files = _client_for(tmp_path).list_files()
    for path in files:
        assert not set(path.split("/")) & set(_IGNORED_DIRS), path


def test_list_files_keeps_a_file_named_like_an_ignored_directory(tmp_path) -> None:
    _build_tree(tmp_path)
    files = _client_for(tmp_path).list_files()
    assert "build.py" in files


def test_list_files_returns_the_exactly_sorted_list(tmp_path) -> None:
    # Built so the expected order cannot come from the walk: ``os.walk`` is
    # top-down, so it always yields this root's own files before descending,
    # yet "a_dir/aa.py" and "a_dir/b.py" sort before the top-level "m.py" and
    # "z.py". A ``list_files`` that forgot ``sorted`` would therefore return a
    # different list whatever order the filesystem enumerates the root in.
    (tmp_path / "m.py").write_text("m = 0\n", encoding="utf-8")
    (tmp_path / "z.py").write_text("z = 0\n", encoding="utf-8")
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "a_dir" / "aa.py").write_text("aa = 0\n", encoding="utf-8")
    (tmp_path / "a_dir" / "b.py").write_text("b = 0\n", encoding="utf-8")
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "b_dir" / "c.py").write_text("c = 0\n", encoding="utf-8")

    files = _client_for(tmp_path).list_files()

    assert files == ["a_dir/aa.py", "a_dir/b.py", "b_dir/c.py", "m.py", "z.py"]


def test_list_files_returns_empty_for_a_missing_directory(tmp_path) -> None:
    files = _client_for(tmp_path).list_files(tmp_path / "missing")
    assert files == []


# --------------------------------------------------------------------------
# complete(): provider dispatch, with no network
# --------------------------------------------------------------------------


async def test_complete_dispatches_to_anthropic() -> None:
    settings = get_settings().model_copy(update={"provider": Provider.ANTHROPIC})
    client = LLMClient(settings=settings)
    client._complete_anthropic = AsyncMock(return_value="a")
    client._complete_openai = AsyncMock(return_value="o")
    client._complete_ollama = AsyncMock(return_value="l")

    assert await client.complete("p") == "a"
    client._complete_anthropic.assert_awaited_once_with("p", "", settings.model)


async def test_complete_dispatches_to_openai() -> None:
    settings = get_settings().model_copy(update={"provider": Provider.OPENAI})
    client = LLMClient(settings=settings)
    client._complete_anthropic = AsyncMock(return_value="a")
    client._complete_openai = AsyncMock(return_value="o")
    client._complete_ollama = AsyncMock(return_value="l")

    assert await client.complete("p") == "o"
    client._complete_openai.assert_awaited_once_with("p", "", settings.model)


async def test_complete_dispatches_to_ollama() -> None:
    settings = get_settings().model_copy(update={"provider": Provider.OLLAMA})
    client = LLMClient(settings=settings)
    client._complete_anthropic = AsyncMock(return_value="a")
    client._complete_openai = AsyncMock(return_value="o")
    client._complete_ollama = AsyncMock(return_value="l")

    assert await client.complete("p") == "l"
    client._complete_ollama.assert_awaited_once_with("p", "", settings.model)


async def test_complete_rejects_an_unknown_provider() -> None:
    # ``model_copy`` does not validate, so a plain bogus string survives here.
    settings = get_settings().model_copy(update={"provider": "nope"})
    client = LLMClient(settings=settings)
    client._complete_anthropic = AsyncMock(return_value="a")
    client._complete_openai = AsyncMock(return_value="o")
    client._complete_ollama = AsyncMock(return_value="l")

    with pytest.raises(ValueError):
        await client.complete("p")

    assert client._complete_anthropic.await_count == 0
    assert client._complete_openai.await_count == 0
    assert client._complete_ollama.await_count == 0


# --------------------------------------------------------------------------
# _complete_anthropic: only text-bearing blocks count
# --------------------------------------------------------------------------


def _client_with_response(response) -> LLMClient:
    client = LLMClient(settings=get_settings())
    create = AsyncMock(return_value=response)
    # Assigning a stub means the property never builds a real client, so no
    # ANTHROPIC_API_KEY is required.
    client._anthropic = SimpleNamespace(messages=SimpleNamespace(create=create))
    return client


async def test_complete_anthropic_skips_non_text_blocks() -> None:
    blocks = [
        SimpleNamespace(type="thinking", thinking="hmm"),
        SimpleNamespace(type="text", text="Hello "),
        SimpleNamespace(type="tool_use", id="t1", name="read_file"),
        SimpleNamespace(type="text", text="world"),
        SimpleNamespace(type="text", text=""),
    ]
    client = _client_with_response(SimpleNamespace(content=blocks))

    assert await client._complete_anthropic("p", "sys", "model-x") == "Hello world"


async def test_complete_anthropic_returns_empty_when_no_block_has_text() -> None:
    blocks = [SimpleNamespace(type="thinking", thinking="hmm")]
    client = _client_with_response(SimpleNamespace(content=blocks))

    assert await client._complete_anthropic("p", "sys", "model-x") == ""


async def test_complete_anthropic_tolerates_a_response_without_content() -> None:
    client = _client_with_response(SimpleNamespace())

    assert await client._complete_anthropic("p", "sys", "model-x") == ""


# --------------------------------------------------------------------------
# _complete_anthropic: a permanent error escapes instead of being retried
# --------------------------------------------------------------------------


def _client_raising(exc: BaseException) -> tuple[LLMClient, list[int]]:
    """Client whose Anthropic stub always raises ``exc``, plus its call counter."""
    calls: list[int] = []

    async def create(**kwargs: object) -> object:
        calls.append(1)
        raise exc

    client = LLMClient(settings=get_settings())
    client._anthropic = SimpleNamespace(messages=SimpleNamespace(create=create))
    return client, calls


async def test_complete_anthropic_does_not_retry_a_permanent_error() -> None:
    """``_is_transient`` gates the retry, and a ``ValueError`` fails that gate.

    ``_RETRY_TRANSIENT`` allows three attempts with ~1s then ~2s of backoff, so
    the single recorded call proves the permanent-error branch: a transient error
    would have called ``create`` three times, and taken seconds to do it.
    """
    client, calls = _client_raising(ValueError("no key"))

    with pytest.raises(ValueError):
        await client._complete_anthropic("p", "sys", "model-x")

    assert len(calls) == 1


# --------------------------------------------------------------------------
# _complete_openai: the chat-completions call, with an injected client
# --------------------------------------------------------------------------


def _openai_response(content: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _client_with_openai(create) -> LLMClient:
    settings = get_settings().model_copy(update={"provider": Provider.OPENAI})
    client = LLMClient(settings=settings)
    # Assigning a stub means the property never builds a real client, so no
    # OPENAI_API_KEY is required.
    client._openai = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client


async def test_complete_openai_returns_the_first_choice_content() -> None:
    captured: dict[str, object] = {}

    async def create(**kwargs: object) -> object:
        captured.update(kwargs)
        return _openai_response("reply")

    client = _client_with_openai(create)

    assert await client._complete_openai("prompt", "sys", "model-x") == "reply"
    assert captured["model"] == "model-x"
    assert captured["max_tokens"] == client.settings.max_tokens
    assert captured["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "prompt"},
    ]


async def test_complete_openai_returns_empty_for_a_null_content() -> None:
    async def create(**kwargs: object) -> object:
        return _openai_response(None)

    client = _client_with_openai(create)

    assert await client._complete_openai("prompt", "sys", "model-x") == ""


# --------------------------------------------------------------------------
# _complete_ollama: the HTTP transport, faked at httpx.AsyncClient
# --------------------------------------------------------------------------


class _FakeOllamaResponse:
    """The two methods ``_complete_ollama`` uses on an httpx response."""

    def __init__(self, data: object) -> None:
        self.data = data
        self.raise_for_status_calls = 0

    def raise_for_status(self) -> None:
        self.raise_for_status_calls += 1

    def json(self) -> object:
        return self.data


class _FakeOllamaClient:
    """Stand-in for ``httpx.AsyncClient``: a context manager with a fake ``post``.

    ``_complete_ollama`` builds its client inside the method body, so the class
    object itself is the only injection point: there is no instance attribute to
    assign beforehand. Each test installs its own subclass (see
    ``_install_fake_ollama``) so recorded calls do not leak between tests.
    """

    instances: list["_FakeOllamaClient"] = []
    response_data: object = None

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout
        self.response = _FakeOllamaResponse(type(self).response_data)
        self.post = AsyncMock(return_value=self.response)
        type(self).instances.append(self)

    async def __aenter__(self) -> "_FakeOllamaClient":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


def _install_fake_ollama(monkeypatch, data: object) -> type["_FakeOllamaClient"]:
    """Replace ``furrow.llm.httpx`` with a fake exposing ``AsyncClient``.

    Returns that client class: its ``instances`` list holds every client the call
    under test built, so ``[]`` means it never opened one.
    """

    class _Client(_FakeOllamaClient):
        response_data = data

    _Client.instances = []
    monkeypatch.setattr("furrow.llm.httpx", SimpleNamespace(AsyncClient=_Client))
    return _Client


def _ollama_client(
    monkeypatch, data: object, base_url: Optional[str] = None
) -> tuple[LLMClient, type]:
    """Install the fake transport, return an Ollama client and its fake class."""
    client_class = _install_fake_ollama(monkeypatch, data)
    update: dict[str, object] = {"provider": Provider.OLLAMA}
    if base_url is not None:
        update["ollama_base_url"] = base_url
    settings = get_settings().model_copy(update=update)
    return LLMClient(settings=settings), client_class


async def test_complete_ollama_posts_the_expected_request(monkeypatch) -> None:
    # The trailing slash is deliberate: the path must be built from the stripped
    # base URL, not from ``.../`` itself.
    client, client_class = _ollama_client(
        monkeypatch, {"message": {"content": "hi"}}, "http://localhost:11434/"
    )

    reply = await client._complete_ollama("prompt", "sys", "model-x")

    assert reply == "hi"
    assert len(client_class.instances) == 1
    http = client_class.instances[0]
    assert http.timeout == _OLLAMA_TIMEOUT_SECONDS
    assert http.post.await_args.args[0] == "http://localhost:11434/api/chat"
    assert http.post.await_args.kwargs["json"] == {
        "model": "model-x",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prompt"},
        ],
        "stream": False,
    }
    assert http.response.raise_for_status_calls == 1


async def test_complete_ollama_returns_empty_without_a_message_key(monkeypatch) -> None:
    client, client_class = _ollama_client(monkeypatch, {"done": True})

    assert await client._complete_ollama("prompt", "sys", "model-x") == ""
    assert len(client_class.instances) == 1


async def test_complete_ollama_returns_empty_for_non_string_content(monkeypatch) -> None:
    client, client_class = _ollama_client(monkeypatch, {"message": {"content": 123}})

    assert await client._complete_ollama("prompt", "sys", "model-x") == ""
    assert len(client_class.instances) == 1
