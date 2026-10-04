"""Tests for :mod:`furrow.llm`: model resolution, response parsing, retries and
the workspace sandbox.

Every provider SDK is replaced with a stub object, so no ``anthropic`` or
``openai`` client is ever constructed and no socket is ever opened. The one
exception is the Ollama path, which talks to an ``http.server`` bound to
``127.0.0.1`` on an ephemeral port.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import furrow.llm as llm_module
from furrow.config import Provider, Settings
from furrow.llm import DEFAULT_SYSTEM_PROMPT, LLMClient, LLMError


class _RecordingMessages:
    """Stub for ``AsyncAnthropic.messages``."""

    def __init__(self, response: Any = None, error: BaseException | None = None) -> None:
        """Store the canned reply or error.

        Args:
            response: Returned by :meth:`create`.
            error: Raised by :meth:`create` when set.
        """
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        """Record the request and produce the canned outcome.

        Args:
            **kwargs: The request payload.

        Returns:
            ``self.response``.

        Raises:
            BaseException: ``self.error`` when one was configured.
        """
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _StubAnthropic:
    """Stand-in for ``AsyncAnthropic`` exposing only ``messages``."""

    def __init__(self, response: Any = None, error: BaseException | None = None) -> None:
        """Build the stub.

        Args:
            response: Value returned by ``messages.create``.
            error: Exception raised by ``messages.create``.
        """
        self.messages = _RecordingMessages(response, error)
        self.kwargs: dict[str, Any] = {}

    async def close(self) -> None:
        """Mirror the SDK coroutine so ``release`` can discard it."""


class _RecordingCompletions:
    """Stub for ``AsyncOpenAI.chat.completions``."""

    def __init__(self, response: Any = None, error: BaseException | None = None) -> None:
        """Store the canned reply or error.

        Args:
            response: Returned by :meth:`create`.
            error: Raised by :meth:`create` when set.
        """
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        """Record the request and produce the canned outcome.

        Args:
            **kwargs: The request payload.

        Returns:
            ``self.response``.

        Raises:
            BaseException: ``self.error`` when one was configured.
        """
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _StubOpenAI:
    """Stand-in for ``AsyncOpenAI`` exposing only ``chat.completions``."""

    def __init__(self, response: Any = None, error: BaseException | None = None) -> None:
        """Build the stub.

        Args:
            response: Value returned by ``chat.completions.create``.
            error: Exception raised by ``chat.completions.create``.
        """
        self.chat = SimpleNamespace(completions=_RecordingCompletions(response, error))

    async def close(self) -> None:
        """Mirror the SDK coroutine so ``release`` can discard it."""


def _install_anthropic(
    monkeypatch: pytest.MonkeyPatch, response: Any = None, error: BaseException | None = None
) -> _StubAnthropic:
    """Swap ``AsyncAnthropic`` for a stub inside :mod:`furrow.llm`.

    Args:
        monkeypatch: Pytest's patcher.
        response: Canned messages response.
        error: Canned messages error.

    Returns:
        The stub, so its recorded calls can be asserted on.
    """
    stub = _StubAnthropic(response, error)
    monkeypatch.setattr(llm_module, "AsyncAnthropic", lambda **_: stub)
    return stub


def _install_openai(
    monkeypatch: pytest.MonkeyPatch, response: Any = None, error: BaseException | None = None
) -> _StubOpenAI:
    """Swap ``AsyncOpenAI`` for a stub inside :mod:`furrow.llm`.

    Args:
        monkeypatch: Pytest's patcher.
        response: Canned chat-completions response.
        error: Canned chat-completions error.

    Returns:
        The stub, so its recorded calls can be asserted on.
    """
    stub = _StubOpenAI(response, error)
    monkeypatch.setattr(llm_module, "AsyncOpenAI", lambda **_: stub)
    return stub


def _text_block(text: str) -> SimpleNamespace:
    """Build a stubbed Anthropic text content block.

    Args:
        text: The block text.

    Returns:
        An object with a ``.text`` attribute.
    """
    return SimpleNamespace(type="text", text=text)


# --------------------------------------------------------------------------- #
# model resolution
# --------------------------------------------------------------------------- #


async def test_complete_uses_the_explicit_model_argument(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    settings = settings_factory(provider=Provider.ANTHROPIC, model="settings-model")
    stub = _install_anthropic(monkeypatch, SimpleNamespace(content=[_text_block("ok")]))

    await LLMClient(settings=settings).complete("hi", model="explicit-model")

    assert stub.messages.calls[0]["model"] == "explicit-model"


async def test_complete_falls_back_to_settings_model(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    settings = settings_factory(provider=Provider.ANTHROPIC, model="settings-model")
    stub = _install_anthropic(monkeypatch, SimpleNamespace(content=[_text_block("ok")]))

    await LLMClient(settings=settings).complete("hi")

    assert stub.messages.calls[0]["model"] == "settings-model"


async def test_override_model_beats_the_explicit_argument(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """This is the ``--model`` contract every agent relies on."""
    settings = settings_factory(
        provider=Provider.ANTHROPIC, model="settings-model", override_model="override-model"
    )
    stub = _install_anthropic(monkeypatch, SimpleNamespace(content=[_text_block("ok")]))

    await LLMClient(settings=settings).complete("hi", model="explicit-model")

    assert stub.messages.calls[0]["model"] == "override-model"


async def test_complete_defaults_the_system_prompt(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    settings = settings_factory(provider=Provider.ANTHROPIC)
    stub = _install_anthropic(monkeypatch, SimpleNamespace(content=[_text_block("ok")]))

    client = LLMClient(settings=settings)
    await client.complete("hi")
    await client.complete("hi", system="be terse")

    assert stub.messages.calls[0]["system"] == DEFAULT_SYSTEM_PROMPT
    assert stub.messages.calls[1]["system"] == "be terse"


# --------------------------------------------------------------------------- #
# missing credentials
# --------------------------------------------------------------------------- #


def test_anthropic_property_without_a_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(settings=Settings(_env_file=None, workspace=Path.cwd()))

    with pytest.raises(LLMError, match="Anthropic API key"):
        _ = client.anthropic


def test_openai_property_without_a_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(settings=Settings(_env_file=None, workspace=Path.cwd()))

    with pytest.raises(LLMError, match="OpenAI API key"):
        _ = client.openai


async def test_complete_wraps_an_unknown_provider(settings_factory: Any) -> None:
    """Pin ``complete``'s unsupported-provider branch, which has no live route.

    ``validate_assignment`` rejects an unknown provider on purpose, so the only way
    to reach ``LLMError: Unsupported provider`` (``furrow/llm.py:245``) is to
    bypass validation deliberately: the branch is a defensive guard, and this is
    the only way to prove it still raises instead of falling through.
    """
    settings = settings_factory()
    # Bypass validation: pydantic would reject this value, but the test needs an
    # unsupported provider to reach the LLMError path.
    object.__setattr__(settings, "provider", "not-a-provider")

    with pytest.raises(LLMError, match="Unsupported provider"):
        await LLMClient(settings=settings).complete("hi")


# --------------------------------------------------------------------------- #
# anthropic response extraction
# --------------------------------------------------------------------------- #


async def test_anthropic_concatenates_every_text_block(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    response = SimpleNamespace(
        content=[
            _text_block("first "),
            SimpleNamespace(type="tool_use", name="bash", input={"cmd": "ls"}),
            _text_block("second"),
        ],
        stop_reason="tool_use",
    )
    _install_anthropic(monkeypatch, response)

    result = await LLMClient(settings=settings_factory()).complete("hi")

    assert result == "first second"


async def test_anthropic_skips_empty_text_blocks(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    response = SimpleNamespace(content=[_text_block(""), _text_block("kept")])
    _install_anthropic(monkeypatch, response)

    assert await LLMClient(settings=settings_factory()).complete("hi") == "kept"


async def test_anthropic_without_text_raises_mentioning_stop_reason(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    response = SimpleNamespace(content=[], stop_reason="max_tokens")
    _install_anthropic(monkeypatch, response)

    with pytest.raises(LLMError) as excinfo:
        await LLMClient(settings=settings_factory()).complete("hi")

    assert "stop_reason" in str(excinfo.value)
    assert "max_tokens" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# openai response extraction
# --------------------------------------------------------------------------- #


async def test_openai_returns_the_first_choice_content(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"), finish_reason="stop")]
    )
    _install_openai(monkeypatch, response)

    result = await LLMClient(
        settings=settings_factory(provider=Provider.OPENAI)
    ).complete("hi")

    assert result == "hello"


async def test_openai_with_no_choices_raises(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    _install_openai(monkeypatch, SimpleNamespace(choices=[]))

    with pytest.raises(LLMError, match="no choices"):
        await LLMClient(settings=settings_factory(provider=Provider.OPENAI)).complete("hi")


async def test_openai_with_null_content_raises(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="length")]
    )
    _install_openai(monkeypatch, response)

    with pytest.raises(LLMError, match="empty content"):
        await LLMClient(settings=settings_factory(provider=Provider.OPENAI)).complete("hi")


# --------------------------------------------------------------------------- #
# ollama, against a loopback-only stub server
# --------------------------------------------------------------------------- #


class _OllamaHandler(BaseHTTPRequestHandler):
    """Serve one canned ``/api/generate`` response and record the request."""

    status = 200
    body: Any = {"response": "ollama says hi"}

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        """Record the payload and answer with the class-level response."""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        self.server.received.append(  # type: ignore[attr-defined]
            {"path": self.path, "payload": json.loads(raw.decode("utf-8"))}
        )
        encoded = json.dumps(type(self).body).encode("utf-8")
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr access log."""


@pytest.fixture
def ollama_server() -> Iterator[ThreadingHTTPServer]:
    """Run a loopback HTTP server that answers ``/api/generate``.

    Yields:
        The running server. Bind host is always ``127.0.0.1`` and the port is
        ephemeral, so no external network is touched.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OllamaHandler)
    server.received = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _ollama_url(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


async def test_ollama_posts_the_documented_payload_and_parses_the_response(
    ollama_server: ThreadingHTTPServer, settings_factory: Any
) -> None:
    url = _ollama_url(ollama_server)
    settings = settings_factory(provider=Provider.OLLAMA, ollama_base_url=url)
    settings.override_model = "llama3"

    result = await LLMClient(settings=settings).complete("hi", system="be terse")

    assert result == "ollama says hi"
    received = ollama_server.received[0]  # type: ignore[attr-defined]
    assert received["path"] == "/api/generate"
    assert received["payload"] == {
        "model": "llama3",
        "prompt": "hi",
        "system": "be terse",
        "stream": False,
    }


async def test_ollama_non_200_raises(
    ollama_server: ThreadingHTTPServer, settings_factory: Any
) -> None:
    _OllamaHandler.status = 400
    _OllamaHandler.body = {"error": "unknown model"}
    url = _ollama_url(ollama_server)
    settings = settings_factory(provider=Provider.OLLAMA, ollama_base_url=url)

    with pytest.raises(LLMError, match="HTTP 400"):
        await LLMClient(settings=settings).complete("hi")


async def test_ollama_server_error_is_wrapped(
    ollama_server: ThreadingHTTPServer, settings_factory: Any
) -> None:
    _OllamaHandler.status = 503
    _OllamaHandler.body = {"error": "busy"}
    url = _ollama_url(ollama_server)
    settings = settings_factory(provider=Provider.OLLAMA, ollama_base_url=url)

    with pytest.raises(LLMError):
        await LLMClient(settings=settings).complete("hi")


async def test_ollama_empty_response_field_raises(
    ollama_server: ThreadingHTTPServer, settings_factory: Any
) -> None:
    _OllamaHandler.status = 200
    _OllamaHandler.body = {"error": "no model loaded"}
    url = _ollama_url(ollama_server)
    settings = settings_factory(provider=Provider.OLLAMA, ollama_base_url=url)

    with pytest.raises(LLMError, match="no text"):
        await LLMClient(settings=settings).complete("hi")


@pytest.fixture(autouse=True)
def _reset_ollama_handler() -> Iterator[None]:
    """Restore the canned Ollama response after each test.

    Yields:
        None.
    """
    yield
    _OllamaHandler.status = 200
    _OllamaHandler.body = {"response": "ollama says hi"}


# --------------------------------------------------------------------------- #
# retry behaviour
# --------------------------------------------------------------------------- #


async def test_transient_failures_are_retried_up_to_max_retries(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    settings = settings_factory(provider=Provider.ANTHROPIC, max_retries=2)
    stub = _install_anthropic(monkeypatch, error=httpx.ConnectError("connection dropped"))

    with pytest.raises(LLMError):
        await LLMClient(settings=settings).complete("hi")

    assert len(stub.messages.calls) == 2


async def test_non_transient_failures_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    settings = settings_factory(provider=Provider.ANTHROPIC, max_retries=3)
    stub = _install_anthropic(monkeypatch, error=ValueError("malformed request"))

    with pytest.raises(LLMError, match="malformed request"):
        await LLMClient(settings=settings).complete("hi")

    assert len(stub.messages.calls) == 1


async def test_a_transient_failure_that_clears_returns_the_retry(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    settings = settings_factory(provider=Provider.ANTHROPIC, max_retries=2)
    stub = _StubAnthropic(SimpleNamespace(content=[_text_block("recovered")]))
    original = stub.messages.create
    attempts = {"count": 0}

    async def create(**kwargs: Any) -> Any:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ReadTimeout("slow")
        return await original(**kwargs)

    stub.messages.create = create  # type: ignore[method-assign]
    monkeypatch.setattr(llm_module, "AsyncAnthropic", lambda **_: stub)

    result = await LLMClient(settings=settings).complete("hi")

    assert result == "recovered"
    assert attempts["count"] == 2


# --------------------------------------------------------------------------- #
# resolve_path: the workspace sandbox
# --------------------------------------------------------------------------- #


def _client(workspace: Path) -> LLMClient:
    """Build a client rooted at ``workspace``.

    Args:
        workspace: The sandbox root.

    Returns:
        A production :class:`LLMClient` — no stubbing, this exercises real code.
    """
    return LLMClient(settings=Settings(_env_file=None, workspace=workspace))


def test_resolve_path_accepts_a_relative_path(tmp_path: Path) -> None:
    resolved = _client(tmp_path).resolve_path("pkg/mod.py")

    assert resolved == (tmp_path / "pkg" / "mod.py").resolve()


def test_resolve_path_accepts_an_absolute_path_inside(tmp_path: Path) -> None:
    target = tmp_path / "inner" / "mod.py"

    assert _client(tmp_path).resolve_path(target) == target.resolve()


def test_resolve_path_rejects_parent_directory_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the workspace"):
        _client(tmp_path).resolve_path("../escape.txt")


def test_resolve_path_rejects_absolute_path_outside(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the workspace"):
        _client(tmp_path).resolve_path("/etc/passwd")


def test_resolve_path_rejects_embedded_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the workspace"):
        _client(tmp_path).resolve_path("pkg/../../escape.txt")


def test_resolve_path_rejects_deep_traversal_that_re_enters(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    with pytest.raises(ValueError, match="outside the workspace"):
        _client(workspace).resolve_path("a/b/../../../outside.txt")


def test_resolve_path_rejects_a_symlinked_directory_pointing_out(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("s3cret", encoding="utf-8")
    (workspace / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside the workspace"):
        _client(workspace).resolve_path("link/secret.txt")


def test_resolve_path_rejects_a_symlinked_file_pointing_out(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("s3cret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside / "secret.txt")

    with pytest.raises(ValueError, match="outside the workspace"):
        _client(workspace).resolve_path("link.txt")


def test_resolve_path_rejects_a_symlinked_parent_of_the_workspace(tmp_path: Path) -> None:
    """A workspace reached through a symlink must still resolve both sides."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    client = _client(link)

    assert client.resolve_path("a.txt") == (real / "a.txt").resolve()


def test_resolve_path_refuses_a_filesystem_root_workspace() -> None:
    client = LLMClient(settings=Settings(_env_file=None, workspace=Path("/")))

    with pytest.raises(ValueError, match="filesystem root"):
        client.resolve_path("anything.txt")


async def test_write_file_refuses_a_filesystem_root_workspace() -> None:
    client = LLMClient(settings=Settings(_env_file=None, workspace=Path("/")))

    with pytest.raises(ValueError, match="filesystem root"):
        await client.write_file("tmp/furrow-should-not-exist.txt", "nope")

    assert not Path("/tmp/furrow-should-not-exist.txt").exists()


# --------------------------------------------------------------------------- #
# write_file / read_file
# --------------------------------------------------------------------------- #


async def test_write_file_creates_missing_parent_directories(tmp_path: Path) -> None:
    client = _client(tmp_path)

    await client.write_file("deep/nested/tree/file.txt", "hello")

    assert (tmp_path / "deep" / "nested" / "tree" / "file.txt").read_text(
        encoding="utf-8"
    ) == "hello"


async def test_write_file_overwrites_an_existing_file(tmp_path: Path) -> None:
    client = _client(tmp_path)
    await client.write_file("a.txt", "first")

    await client.write_file("a.txt", "second")

    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "second"


async def test_write_file_refuses_escapes_and_creates_nothing(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    client = _client(workspace)

    with pytest.raises(ValueError, match="outside the workspace"):
        await client.write_file("../escaped.txt", "nope")

    assert not (tmp_path / "escaped.txt").exists()


async def test_write_file_refuses_absolute_escapes(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = tmp_path / "escaped.txt"
    client = _client(workspace)

    with pytest.raises(ValueError, match="outside the workspace"):
        await client.write_file(target, "nope")

    assert not target.exists()


async def test_read_file_round_trips_a_write_file(tmp_path: Path) -> None:
    client = _client(tmp_path)
    await client.write_file("pkg/module.py", "# hello\nprint('hi')\n")

    assert await client.read_file("pkg/module.py") == "# hello\nprint('hi')\n"


async def test_read_file_refuses_escapes(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (tmp_path / "secret.txt").write_text("s3cret", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the workspace"):
        await _client(workspace).read_file("../secret.txt")


async def test_read_file_accepts_an_absolute_path_inside(tmp_path: Path) -> None:
    target = tmp_path / "inside.txt"
    target.write_text("content", encoding="utf-8")

    assert await _client(tmp_path).read_file(target) == "content"


async def test_read_file_missing_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        await _client(tmp_path).read_file("nope.txt")


# --------------------------------------------------------------------------- #
# list_files
# --------------------------------------------------------------------------- #


def _populate(root: Path) -> None:
    """Create a small noisy project tree under ``root``.

    Args:
        root: Directory to populate.
    """
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (root / "top.py").write_text("y = 2\n", encoding="utf-8")
    for noisy in (
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
    ):
        (root / noisy).mkdir()
        (root / noisy / "junk.py").write_text("junk\n", encoding="utf-8")


def test_list_files_returns_sorted_relative_posix_paths(tmp_path: Path) -> None:
    _populate(tmp_path)

    assert _client(tmp_path).list_files() == ["pkg/mod.py", "top.py"]


def test_list_files_prunes_noisy_directories(tmp_path: Path) -> None:
    _populate(tmp_path)

    listed = _client(tmp_path).list_files()

    assert listed
    assert not any("junk.py" in path for path in listed)


def test_list_files_does_not_descend_into_symlinked_directories(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "inner.py").write_text("z = 3\n", encoding="utf-8")
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)

    assert _client(tmp_path).list_files() == ["real/inner.py"]


def test_list_files_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert _client(tmp_path).list_files(tmp_path / "does-not-exist") == []


def test_list_files_of_a_file_is_empty(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_text("x\n", encoding="utf-8")

    assert _client(tmp_path).list_files(target) == []


def test_list_files_rejects_an_escaping_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the workspace"):
        _client(tmp_path / "ws").list_files("..")


def test_list_files_is_capped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for index in range(12):
        (tmp_path / f"file{index:02d}.txt").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(llm_module, "MAX_LIST_FILES", 5)

    assert _client(tmp_path).list_files() == [f"file{index:02d}.txt" for index in range(5)]


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def test_release_is_idempotent_and_safe(settings_factory: Any) -> None:
    client = _client(settings_factory().workspace)

    client.release()
    client.release()


def test_release_drops_the_cached_sdk_clients(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    stub = _install_anthropic(monkeypatch, SimpleNamespace(content=[_text_block("ok")]))
    client = LLMClient(settings=settings_factory())
    _ = client.anthropic
    assert client._anthropic is stub

    client.release()

    assert client._anthropic is None
