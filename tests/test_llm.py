"""Regression tests for :mod:`furrow.llm`.

Covers JSON recovery from messy model output, the retry/backoff policy, provider routing,
tolerant response parsing, and the filesystem helpers. Every provider call is intercepted by
a fake client object, so nothing here can reach the network.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any

import pytest

from furrow.config import Provider, Settings
from furrow.llm import (
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_SETTINGS,
    RETRYABLE_STATUS_CODES,
    LLMClient,
    _anthropic_text,
    _openai_text,
    extract_json,
)


class HttpError(Exception):
    """A stand-in for a provider SDK error carrying an HTTP status code."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"http {status_code}")


def anthropic_response(text: str = "hello") -> Any:
    return types.SimpleNamespace(content=[types.SimpleNamespace(text=text)])


def openai_response(content: Any = "hello") -> Any:
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


class FakeAnthropic:
    """Replaces ``AsyncAnthropic``; pops queued exceptions then returns canned text."""

    def __init__(
        self, errors: list[BaseException | None] | None = None, text: str = "hello"
    ) -> None:
        # ``None`` marks the point at which the fake starts succeeding.
        self.errors = [error for error in (errors or []) if error is not None]
        self.text = text
        self.calls = 0
        self.kwargs: dict[str, Any] | None = None
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.kwargs = kwargs
        if self.errors:
            raise self.errors.pop(0)
        return anthropic_response(self.text)


class FakeOpenAI:
    """Replaces ``AsyncOpenAI`` for both the openai and ollama routes."""

    def __init__(
        self, errors: list[BaseException | None] | None = None, content: Any = "hello"
    ) -> None:
        self.errors = [error for error in (errors or []) if error is not None]
        self.content = content
        self.calls = 0
        self.kwargs: dict[str, Any] | None = None
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.kwargs = kwargs
        if self.errors:
            raise self.errors.pop(0)
        return openai_response(self.content)


def anthropic_client(
    monkeypatch: pytest.MonkeyPatch,
    errors: list[BaseException | None] | None = None,
    text: str = "hello",
    **settings_kwargs: Any,
) -> LLMClient:
    settings = Settings(anthropic_api_key="sk-ant-test", **settings_kwargs)
    client = LLMClient(settings=settings)
    client._anthropic = FakeAnthropic(errors, text)
    return client


def record_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``asyncio.sleep`` so backoff is instant but still observable.

    Returns the list every replaced sleep appends its requested delay to, which is how the
    ``backoff * 2 ** attempt`` progression is asserted without spending real seconds.
    """
    delays: list[float] = []

    async def fake_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return delays


# --------------------------------------------------------------------------- extract_json


@pytest.mark.parametrize(
    ("label", "text", "expected"),
    [
        ("plain object", '{"a": 1}', {"a": 1}),
        (
            "json fenced",
            'Here is the plan:\n```json\n{"tasks": [{"id": "1"}]}\n```\nDone.',
            {"tasks": [{"id": "1"}]},
        ),
        (
            "fenced without a language tag",
            '```\n{"a": 1}\n```',
            {"a": 1},
        ),
        (
            "prose wrapped",
            'Sure! {"a": 1} Hope that helps.',
            {"a": 1},
        ),
        ("trailing comma", '{"a": 1,}', {"a": 1}),
        ("trailing comma nested", '{"a": 1, "b": [1, 2,],}', {"a": 1, "b": [1, 2]}),
        (
            "nested trailing comma in a fenced block",
            '```json\n{\n  "a": 1,\n  "b": [1, 2,],\n}\n```',
            {"a": 1, "b": [1, 2]},
        ),
        ("closing brace inside a string", '{"a": "}"}', {"a": "}"}),
        ("opening brace inside a string", '{"a": "{"}', {"a": "{"}),
        ("escaped quotes inside a string", '{"a": "he said \\"hi\\""}', {"a": 'he said "hi"'}),
        (
            "escaped backslash then quote",
            '{"a": "back\\\\ then \\"quote\\""}',
            {"a": 'back\\ then "quote"'},
        ),
        ("null and true", '{"a": null, "b": true}', {"a": None, "b": True}),
        (
            "false and nested array",
            '{"a": false, "b": [[1], [2, 3]]}',
            {"a": False, "b": [[1], [2, 3]]},
        ),
        ("top level array", "[1, 2, 3]", [1, 2, 3]),
        ("top level array with a trailing comma", "[1, 2, 3,]", [1, 2, 3]),
        ("top level string", '"just a string"', "just a string"),
        ("top level number", "42", 42),
        ("python literals", '{"a": True, "b": None}', {"a": True, "b": None}),
        ("empty object", "{}", {}),
    ],
)
def test_extract_json_recovers_json_from_model_noise(
    label: str, text: str, expected: object
) -> None:
    assert extract_json(text) == expected, label


@pytest.mark.parametrize(
    "text",
    [
        "I am afraid I cannot help with that.",
        "",
        "   ",
        "```\nnot json at all\n```",
        "{unbalanced: [1, 2}",
    ],
)
def test_extract_json_raises_value_error_on_unusable_output(text: str) -> None:
    with pytest.raises(ValueError, match="Could not extract JSON"):
        extract_json(text)


def test_extract_json_error_message_truncates_a_huge_response() -> None:
    with pytest.raises(ValueError) as excinfo:
        extract_json("x" * 5000)

    message = str(excinfo.value)
    assert "..." in message
    assert len(message) < 1000


def test_extract_json_rejects_non_string_input() -> None:
    with pytest.raises(TypeError):
        extract_json(None)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------ retries


def test_transient_errors_are_retried_then_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = record_sleeps(monkeypatch)
    client = anthropic_client(
        monkeypatch,
        errors=[HttpError(429), HttpError(500), HttpError(503)],
        llm_max_retries=3,
        llm_backoff_seconds=0.5,
    )

    async def scenario() -> str:
        return await client.complete("prompt")

    assert asyncio.run(scenario()) == "hello"
    # llm_max_retries counts *extra* attempts, so 3 means four calls in total.
    assert client._anthropic.calls == 4
    assert delays == [0.5, 1.0, 2.0]


def test_retries_are_bounded_then_the_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = record_sleeps(monkeypatch)
    client = anthropic_client(
        monkeypatch, errors=[HttpError(429) for _ in range(10)], llm_max_retries=2
    )

    async def scenario() -> None:
        await client.complete("prompt")

    with pytest.raises(HttpError):
        asyncio.run(scenario())
    assert client._anthropic.calls == 3
    assert len(delays) == 2


def test_zero_retries_means_a_single_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = record_sleeps(monkeypatch)
    client = anthropic_client(monkeypatch, errors=[HttpError(500)], llm_max_retries=0)

    async def scenario() -> None:
        await client.complete("prompt")

    with pytest.raises(HttpError):
        asyncio.run(scenario())
    assert client._anthropic.calls == 1
    assert delays == []


@pytest.mark.parametrize("status_code", [401, 403, 400, 404, 422])
def test_auth_and_other_client_errors_are_not_retried(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    delays = record_sleeps(monkeypatch)
    client = anthropic_client(monkeypatch, errors=[HttpError(status_code), None], llm_max_retries=3)

    async def scenario() -> None:
        await client.complete("prompt")

    with pytest.raises(HttpError):
        asyncio.run(scenario())
    assert client._anthropic.calls == 1
    assert delays == []


@pytest.mark.parametrize("status_code", sorted(RETRYABLE_STATUS_CODES))
def test_every_documented_retryable_status_is_retried(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    record_sleeps(monkeypatch)
    client = anthropic_client(monkeypatch, errors=[HttpError(status_code), None], llm_max_retries=1)

    async def scenario() -> str:
        return await client.complete("prompt")

    assert asyncio.run(scenario()) == "hello"
    assert client._anthropic.calls == 2


def test_cancellation_is_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = record_sleeps(monkeypatch)
    client = anthropic_client(monkeypatch, errors=[asyncio.CancelledError(), None])

    async def scenario() -> None:
        await client.complete("prompt")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())
    assert client._anthropic.calls == 1
    assert delays == []


def test_connection_errors_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    record_sleeps(monkeypatch)
    client = anthropic_client(
        monkeypatch, errors=[ConnectionResetError("boom"), None], llm_max_retries=1
    )

    async def scenario() -> str:
        return await client.complete("prompt")

    assert asyncio.run(scenario()) == "hello"
    assert client._anthropic.calls == 2


def test_unclassifiable_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    record_sleeps(monkeypatch)
    client = anthropic_client(
        monkeypatch, errors=[TypeError("bad request shape"), None], llm_max_retries=3
    )

    async def scenario() -> None:
        await client.complete("prompt")

    with pytest.raises(TypeError):
        asyncio.run(scenario())
    assert client._anthropic.calls == 1


def test_backoff_settings_are_clamped() -> None:
    assert LLMClient(settings=Settings(llm_max_retries=-5)).max_retries == 0
    assert LLMClient(settings=Settings(llm_backoff_seconds=-1)).backoff_seconds == 0.0


def test_backoff_settings_fall_back_when_unusable() -> None:
    class Odd:
        llm_max_retries = "not a number"
        llm_backoff_seconds = "not a number"

    client = LLMClient(settings=Settings())
    client.settings = Odd()  # type: ignore[assignment]

    assert client.max_retries == 3
    assert client.backoff_seconds == 1.0


# --------------------------------------------------------------------------- routing


def test_anthropic_provider_dispatches_to_the_anthropic_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = anthropic_client(monkeypatch, text="from anthropic")

    async def scenario() -> str:
        return await client.complete("prompt", "be terse", model="model-x")

    assert asyncio.run(scenario()) == "from anthropic"
    assert client._anthropic.calls == 1
    assert client._anthropic.kwargs == {
        "model": "model-x",
        "max_tokens": 4096,
        "system": "be terse",
        "messages": [{"role": "user", "content": "prompt"}],
    }


def test_openai_provider_dispatches_to_the_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(provider=Provider.OPENAI, openai_api_key="sk-openai-test")
    client = LLMClient(settings=settings)
    fake = FakeOpenAI(content="from openai")
    client._openai = fake

    async def scenario() -> str:
        return await client.complete("prompt", "be terse", model="gpt-x")

    assert asyncio.run(scenario()) == "from openai"
    assert client.chat_client is fake
    assert fake.kwargs == {
        "model": "gpt-x",
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "prompt"},
        ],
    }


def test_ollama_provider_dispatches_to_the_openai_compatible_client() -> None:
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://box:1234")
    client = LLMClient(settings=settings)
    fake = FakeOpenAI(content="from ollama")
    client._ollama = fake

    async def scenario() -> str:
        return await client.complete("prompt", model="llama-x")

    assert asyncio.run(scenario()) == "from ollama"
    assert client.chat_client is fake
    assert fake.kwargs["model"] == "llama-x"


@pytest.mark.parametrize(
    ("configured", "expected_path"),
    [
        ("http://box:1234", "/v1/"),
        ("http://box:1234/", "/v1/"),
        ("http://box:1234/v1", "/v1/"),
        ("http://box:1234/v1/", "/v1/"),
        ("  http://box:1234///  ", "/v1/"),
        ("", "/v1/"),
        ("   ", "/v1/"),
    ],
)
def test_ollama_base_url_is_normalised(configured: str, expected_path: str) -> None:
    settings = Settings(provider=Provider.OLLAMA, ollama_base_url=configured)
    client = LLMClient(settings=settings)

    base_url = str(client.ollama.base_url)

    assert base_url.endswith(expected_path), base_url


def test_ollama_empty_base_url_falls_back_to_the_documented_default() -> None:
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA, ollama_base_url=""))

    assert str(client.ollama.base_url).startswith("http://localhost:11434/v1")


def test_ollama_needs_no_api_key() -> None:
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA, anthropic_api_key=None))
    client.settings.anthropic_api_key = None
    client.settings.openai_api_key = None

    assert client.ollama is not None


@pytest.mark.parametrize(
    ("provider", "missing_env"),
    [(Provider.ANTHROPIC, "ANTHROPIC_API_KEY"), (Provider.OPENAI, "OPENAI_API_KEY")],
)
def test_a_missing_api_key_raises_with_the_env_var_name(
    monkeypatch: pytest.MonkeyPatch, provider: Provider, missing_env: str
) -> None:
    monkeypatch.delenv(missing_env, raising=False)
    client = LLMClient(settings=Settings(provider=provider))

    with pytest.raises(ValueError, match=missing_env):
        _ = client.anthropic if provider is Provider.ANTHROPIC else client.openai


def test_unprefixed_env_var_is_used_when_the_setting_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))

    assert client.anthropic is not None


def test_the_furrow_prefixed_key_wins_over_the_unprefixed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-unprefixed")
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "from-prefixed")
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))

    assert client.anthropic is not None
    assert client.settings.anthropic_api_key == "from-prefixed"


# ------------------------------------------------------------------- response parsing


@pytest.mark.parametrize(
    "response",
    [
        types.SimpleNamespace(content=[]),
        types.SimpleNamespace(content=None),
        types.SimpleNamespace(),
        types.SimpleNamespace(content=[types.SimpleNamespace()]),
        types.SimpleNamespace(content="not a list of blocks"),
    ],
)
def test_anthropic_text_tolerates_unusable_content(response: Any) -> None:
    assert _anthropic_text(response) == ""


def test_anthropic_text_joins_every_text_block() -> None:
    response = types.SimpleNamespace(
        content=[
            types.SimpleNamespace(text="one "),
            types.SimpleNamespace(no_text=True),
            types.SimpleNamespace(text="two"),
        ]
    )

    assert _anthropic_text(response) == "one two"


@pytest.mark.parametrize(
    "response",
    [
        types.SimpleNamespace(choices=[]),
        types.SimpleNamespace(choices=None),
        types.SimpleNamespace(),
        types.SimpleNamespace(choices=[types.SimpleNamespace(message=None)]),
        types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace())]),
        types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=None))]
        ),
    ],
)
def test_openai_text_tolerates_unusable_content(response: Any) -> None:
    assert _openai_text(response) == ""


def test_openai_text_joins_content_parts() -> None:
    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=[types.SimpleNamespace(text="one "), types.SimpleNamespace(text="two")]
                )
            )
        ]
    )

    assert _openai_text(response) == "one two"


def test_complete_returns_empty_string_for_an_empty_anthropic_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = anthropic_client(monkeypatch, text="")
    client._anthropic.errors = []
    original = client._anthropic.create

    async def empty(**kwargs: Any) -> Any:
        await original(**kwargs)
        return types.SimpleNamespace(content=[])

    client._anthropic.create = empty  # type: ignore[method-assign]

    async def scenario() -> str:
        return await client.complete("prompt")

    assert asyncio.run(scenario()) == ""


# ------------------------------------------------------------------- file helpers


def build_tree(root: Any) -> None:
    paths = [
        "a.py",
        "b/c.py",
        "z/y/w.py",
        ".hidden/d.py",
        ".dotfile",
        "node_modules/e.js",
        "__pycache__/f.pyc",
        "venv/g.py",
        "dist/h.js",
        "build/i.py",
        "site-packages/j.py",
        "target/k.py",
        "vendor/l.py",
        "htmlcov/m.js",
    ]
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")


def test_list_files_prunes_hidden_and_vendored_directories(tmp_path: Any) -> None:
    build_tree(tmp_path)

    listed = LLMClient(settings=Settings()).list_files(tmp_path)

    assert ".hidden/d.py" not in listed
    assert not any(part.startswith(".") and "/" in part for part in listed)
    for name in DEFAULT_EXCLUDED_DIRS:
        assert not any(part.startswith(f"{name}/") for part in listed), name
    assert "a.py" in listed
    assert "b/c.py" in listed
    assert "z/y/w.py" in listed
    # Hidden *files* are context-worth keeping; only hidden *directories* are pruned.
    assert ".dotfile" in listed


def test_list_files_is_sorted(tmp_path: Any) -> None:
    build_tree(tmp_path)

    listed = LLMClient(settings=Settings()).list_files(tmp_path)

    assert listed == sorted(listed)


@pytest.mark.parametrize(("max_files", "expected"), [(1, 1), (3, 3), (None, 4), (0, 4)])
def test_list_files_respects_max_files(tmp_path: Any, max_files: int | None, expected: int) -> None:
    build_tree(tmp_path)

    listed = LLMClient(settings=Settings()).list_files(tmp_path, max_files=max_files)

    assert len(listed) == expected


def test_list_files_honours_extra_exclusions(tmp_path: Any) -> None:
    build_tree(tmp_path)

    listed = LLMClient(settings=Settings()).list_files(tmp_path, exclude=("b", "z"))

    assert not any(part.startswith("b/") for part in listed)
    assert not any(part.startswith("z/") for part in listed)
    assert "a.py" in listed


def test_list_files_returns_empty_for_a_missing_directory(tmp_path: Any) -> None:
    client = LLMClient(settings=Settings())

    assert client.list_files(tmp_path / "does-not-exist") == []


def test_list_files_returns_empty_for_a_file_path(tmp_path: Any) -> None:
    target = tmp_path / "a.py"
    target.write_text("x", encoding="utf-8")

    assert LLMClient(settings=Settings()).list_files(target) == []


def test_read_file_round_trips(tmp_path: Any) -> None:
    target = tmp_path / "note.txt"
    target.write_text("hello\nworld\n", encoding="utf-8")

    async def scenario() -> str:
        return await LLMClient(settings=Settings()).read_file(target)

    assert asyncio.run(scenario()) == "hello\nworld\n"


def test_read_file_raises_for_a_missing_path(tmp_path: Any) -> None:
    async def scenario() -> None:
        await LLMClient(settings=Settings()).read_file(tmp_path / "nope.txt")

    with pytest.raises(FileNotFoundError, match="nope.txt"):
        asyncio.run(scenario())


def test_read_file_raises_for_a_directory(tmp_path: Any) -> None:
    (tmp_path / "sub").mkdir()

    async def scenario() -> None:
        await LLMClient(settings=Settings()).read_file(tmp_path / "sub")

    with pytest.raises(IsADirectoryError):
        asyncio.run(scenario())


def test_write_file_creates_parents(tmp_path: Any) -> None:
    target = tmp_path / "deep" / "nested" / "out.txt"

    async def scenario() -> None:
        await LLMClient(settings=Settings()).write_file(target, "written")

    asyncio.run(scenario())

    assert target.read_text(encoding="utf-8") == "written"


# ------------------------------------------------------------------------- constructor


def test_settings_none_falls_back_to_the_module_singleton() -> None:
    client = LLMClient(settings=None)

    assert client.settings is not None
    assert client.settings is DEFAULT_SETTINGS


def test_settings_positional_argument_is_honoured() -> None:
    settings = Settings(model="custom-model")

    assert LLMClient(settings).settings is settings


def test_clients_are_cached_across_property_accesses(monkeypatch: pytest.MonkeyPatch) -> None:
    client = anthropic_client(monkeypatch)

    assert client.anthropic is client.anthropic


def test_complete_raises_for_an_unsupported_provider() -> None:
    class Fake:
        provider = "telepathy"
        model = "x"

    client = LLMClient(settings=Settings())
    client.settings = Fake()  # type: ignore[assignment]

    async def scenario() -> None:
        await client.complete("prompt")

    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(scenario())
