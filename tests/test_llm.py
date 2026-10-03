"""Offline tests for :mod:`furrow.llm`.

The suite never touches the network. Provider traffic is asserted either on the
*configured endpoint* (an :class:`openai.AsyncOpenAI` renders its ``base_url``
without connecting) or on a stub injected in place of the lazily built SDK
client. Missing credentials are a first-class case, not an accident: the shared
autouse ``_no_api_keys`` fixture scrubs every key variable, so a test that
touches :attr:`LLMClient.anthropic` is guaranteed to fail fast instead of
dialling out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import furrow.config
from furrow.config import Provider, Settings
from furrow.llm import (
    DEFAULT_SYSTEM_PROMPT,
    LLMClient,
    _anthropic_text,
    _openai_text,
    extract_json,
)

if TYPE_CHECKING:
    from collections.abc import Callable

#: Canned body the stubbed Ollama endpoint returns, distinct from any real reply.
OLLAMA_REPLY = "ollama stub reply"

#: Directories/suffixes :meth:`LLMClient.list_files` must never surface.
JUNK_PATHS: tuple[str, ...] = (
    "__pycache__/main.cpython-311.pyc",
    ".git/config",
    ".venv/lib/site.py",
    "node_modules/left-pad/index.js",
    "orphan.pyc",
)

#: Files that must survive pruning, in the order ``list_files`` reports them.
KEPT_PATHS: tuple[str, ...] = ("README.md", "main.py", "pkg/__init__.py", "pkg/util.py")

#: Every file written by :func:`_make_tree`, created in deliberately unsorted
#: order so the sorting assertion cannot pass by accident.
TREE_FILES: tuple[str, ...] = (
    "pkg/util.py",
    "main.py",
    "node_modules/left-pad/index.js",
    "scratch/notes.md",
    "README.md",
    "__pycache__/main.cpython-311.pyc",
    "orphan.pyc",
    "pkg/__init__.py",
    ".venv/lib/site.py",
    ".git/config",
)


def _make_tree(root: Path) -> Path:
    """Create a small project tree containing both signal files and junk."""
    for relative in TREE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return root


# --------------------------------------------------------------- extract_json
def test_extract_json_returns_bare_object_unchanged() -> None:
    # Baseline: a well-behaved model reply must round-trip byte-identically.
    assert json.loads(extract_json('{"a": 1}')) == {"a": 1}


def test_extract_json_unwraps_json_code_fence() -> None:
    # Regression: fenced JSON used to reach json.loads() verbatim and raise,
    # which killed the whole cycle instead of just the one parse.
    reply = 'Here you go:\n\n```json\n{"tasks": ["one"]}\n```\n'
    assert json.loads(extract_json(reply)) == {"tasks": ["one"]}


def test_extract_json_skips_prose_preamble() -> None:
    # Models habitually chat before answering; the preamble is not JSON.
    reply = 'Here is the plan you asked for:\n\n{"rationale": "small"}\n'
    assert json.loads(extract_json(reply)) == {"rationale": "small"}


def test_extract_json_skips_trailing_prose() -> None:
    # The mirror image: commentary after the payload must not be swallowed.
    reply = '{"a": 1}\n\nLet me know if you want a different breakdown.'
    assert json.loads(extract_json(reply)) == {"a": 1}


def test_extract_json_keeps_braces_inside_string_values() -> None:
    # A naive brace counter truncated the payload at the inner "}", silently
    # dropping the rest of the object; string interiors must be skipped.
    reply = '{"note": "use { and } carefully"}'
    extracted = extract_json(reply)
    assert extracted == reply
    assert json.loads(extracted) == {"note": "use { and } carefully"}


def test_extract_json_repairs_trailing_comma() -> None:
    # Trailing commas are invalid JSON but extremely common model output.
    assert json.loads(extract_json('{"a": 1, "b": 2,}')) == {"a": 1, "b": 2}


def test_extract_json_accepts_top_level_array() -> None:
    # Reviewer-style replies are arrays, not objects; both shapes are valid.
    assert json.loads(extract_json('[{"id": "1"}, {"id": "2"}]')) == [
        {"id": "1"},
        {"id": "2"},
    ]


def test_extract_json_round_trips_nested_structure() -> None:
    # Depth matters: the balanced scanner must not stop at the first inner "}".
    payload = {
        "tasks": [{"id": "1", "files": ["a.py"], "dependencies": []}],
        "rationale": "keep the diff small",
    }
    assert json.loads(extract_json(json.dumps(payload))) == payload


def test_extract_json_rejects_text_without_json() -> None:
    # No JSON at all is a hard error, not a silent empty plan.
    with pytest.raises(ValueError, match="no JSON object or array found"):
        extract_json("The model refused to answer that question.")


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_extract_json_rejects_empty_or_blank_input(blank: str) -> None:
    # An empty reply must raise rather than return "" and blow up downstream.
    with pytest.raises(ValueError, match="empty response, no JSON found"):
        extract_json(blank)


# ---------------------------------------------------------------- list_files
def test_list_files_prunes_junk_directories_and_bytecode(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    # Junk swamping the prompt wastes tokens and can leak secrets, so the walk
    # must never descend into __pycache__/.git/.venv/node_modules.
    root = _make_tree(tmp_path / "project")
    files = LLMClient(settings=make_settings()).list_files(root)

    assert sorted(files) == [*KEPT_PATHS, "scratch/notes.md"]
    for junk in JUNK_PATHS:
        assert junk not in files
    prefixes = (".git", ".venv", "node_modules", "__pycache__")
    assert not [f for f in files if f.startswith(prefixes)]


def test_list_files_returns_paths_relative_to_the_given_directory(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    # Callers join the results onto their own root, so absolute or root-prefixed
    # paths would produce garbage double paths in every prompt.
    root = _make_tree(tmp_path / "project")
    files = LLMClient(settings=make_settings()).list_files(root)

    assert "pkg/util.py" in files
    assert all(not Path(f).is_absolute() for f in files)
    assert not [f for f in files if f.startswith(str(root))]


def test_list_files_output_is_sorted(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    # Determinism matters for reproducible prompts and cache hits; the tree is
    # written in unsorted order, so this cannot pass by filesystem order luck.
    root = _make_tree(tmp_path / "project")
    files = LLMClient(settings=make_settings()).list_files(root)

    assert files == sorted(files)
    assert len(files) == len(set(files))


def test_list_files_honours_limit(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    # A huge repo must not be able to overflow the prompt budget.
    root = tmp_path / "flat"
    for name in ("d.py", "a.py", "c.py", "b.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x", encoding="utf-8")

    client = LLMClient(settings=make_settings())
    assert client.list_files(root, limit=2) == ["a.py", "b.py"]
    assert client.list_files(root, limit=0) == []
    assert len(client.list_files(root)) == 4


def test_list_files_missing_directory_returns_empty(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    # A not-yet-created workspace is a normal state, not an exception.
    client = LLMClient(settings=make_settings())
    assert client.list_files(tmp_path / "does-not-exist") == []


def test_list_files_honours_extra_ignore_entry(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    # Callers exclude their own noise (local notes, fixtures) on top of the
    # built-in junk list.
    root = _make_tree(tmp_path / "project")
    client = LLMClient(settings=make_settings())

    with_default = client.list_files(root)
    with_ignored = client.list_files(root, ignore=["scratch"])

    assert "scratch/notes.md" in with_default
    assert "scratch/notes.md" not in with_ignored
    assert with_ignored == [f for f in with_default if f != "scratch/notes.md"]


# ------------------------------------------------- construction, no credentials
def test_client_constructs_without_any_api_key() -> None:
    # Regression: eager client construction in __init__ made a missing key a
    # hard failure at startup; the SDKs must stay lazily built.
    client = LLMClient()

    assert client.settings is not None
    assert client._anthropic is None
    assert client._openai is None
    assert client._ollama is None


def test_client_with_settings_none_falls_back_to_module_default() -> None:
    # Agents hold an optional Settings and pass it straight through, so None
    # must mean "use the singleton", not raise.
    assert LLMClient(settings=None).settings is furrow.config.settings


def test_anthropic_property_raises_when_key_is_missing(
    make_settings: Callable[..., Settings],
) -> None:
    # The error must name the missing variable so a user can fix their env; it
    # is raised locally and never reaches the network.
    client = LLMClient(settings=make_settings(anthropic_api_key=None))

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = client.anthropic


# ------------------------------------------------------------- ollama endpoint
async def test_ollama_base_url_ending_in_v1_is_not_doubled(
    make_settings: Callable[..., Settings],
) -> None:
    # Users paste the full OpenAI-compatible endpoint; a naive f"{base}/v1"
    # produced /v1/v1 and every request 404'd. Constructing AsyncOpenAI renders
    # base_url without connecting.
    client = LLMClient(settings=make_settings(ollama_base_url="http://ollama.internal:11434/v1"))

    assert client._ollama_base_url == "http://ollama.internal:11434"
    assert str(client.ollama.base_url) == "http://ollama.internal:11434/v1/"
    assert "/v1/v1" not in str(client.ollama.base_url)

    await client.close()


async def test_ollama_base_url_trailing_slashes_and_whitespace_normalise(
    make_settings: Callable[..., Settings],
) -> None:
    # Copy-pasted URLs arrive padded and slash-terminated; both must collapse to
    # the same canonical origin.
    client = LLMClient(settings=make_settings(ollama_base_url="  http://h:11434///  "))

    assert client._ollama_base_url == "http://h:11434"
    assert str(client.ollama.base_url) == "http://h:11434/v1/"

    await client.close()


def test_ollama_empty_base_url_raises_value_error(
    make_settings: Callable[..., Settings],
) -> None:
    # An unset endpoint is a configuration error worth naming, not a request to
    # the relative URL "/v1".
    client = LLMClient(settings=make_settings(ollama_base_url=""))

    with pytest.raises(ValueError, match="ollama_base_url is not set"):
        _ = client._ollama_base_url


# ----------------------------------------------------------------- ollama stub
@dataclass
class _StubMessage:
    content: str | None


@dataclass
class _StubChoice:
    message: _StubMessage
    finish_reason: str = "stop"


@dataclass
class _StubResponse:
    choices: list[_StubChoice]


@dataclass
class _StubCompletions:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        return _StubResponse(choices=[_StubChoice(message=_StubMessage(content=OLLAMA_REPLY))])


@dataclass
class _StubChat:
    completions: _StubCompletions = field(default_factory=_StubCompletions)


@dataclass
class _StubOllama:
    chat: _StubChat = field(default_factory=_StubChat)


async def test_complete_dispatches_ollama_instead_of_rejecting_the_provider(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: Provider.OLLAMA was missing from the dispatch table, so
    # complete() raised "Unsupported provider". The stub replaces the lazily
    # built SDK client, so the assertion is on what reached the provider.
    client = LLMClient(
        settings=make_settings(provider=Provider.OLLAMA, model="llama3.2", ollama_base_url="")
    )
    stub = _StubOllama()
    monkeypatch.setattr(client, "_ollama", stub)

    reply = await client.complete("summarise the repo", system="be terse", model="qwen2.5")

    assert reply == OLLAMA_REPLY
    call = stub.chat.completions.calls[0]
    assert call["model"] == "qwen2.5"
    assert call["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "summarise the repo"},
    ]


async def test_complete_falls_back_to_settings_model_for_ollama(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A locally hosted model is named in settings only; model=None must resolve
    # to settings.model rather than sending None to the endpoint.
    settings = make_settings(provider=Provider.OLLAMA, model="llama3.2", ollama_base_url="")
    client = LLMClient(settings=settings)
    stub = _StubOllama()
    monkeypatch.setattr(client, "_ollama", stub)

    await client.complete("hello")

    call = stub.chat.completions.calls[0]
    assert call["model"] == "llama3.2"
    # An empty system prompt still needs a usable instruction for the model.
    assert call["messages"][0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}


# -------------------------------------------------------- response hardening
def test_anthropic_text_joins_text_blocks() -> None:
    # The happy path stays intact after the hardening below.
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="one "),
            SimpleNamespace(type="text", text="two"),
        ]
    )
    assert _anthropic_text(response) == "one two"


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(content=[]),
        SimpleNamespace(content=None),
        SimpleNamespace(),
        SimpleNamespace(content=[SimpleNamespace(type="tool_use", id="t1")]),
    ],
    ids=["empty", "null-content", "missing-attr", "non-text-block"],
)
def test_anthropic_text_rejects_responses_without_text(response: Any) -> None:
    # The old code did response.content[0].text, which raised IndexError on a
    # blocked/empty reply; the helper now fails with an actionable ValueError.
    with pytest.raises(ValueError, match="no text content block"):
        _anthropic_text(response)


def test_openai_text_returns_first_message_content() -> None:
    # The happy path stays intact after the hardening below.
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
    assert _openai_text(response) == "hi"


@pytest.mark.parametrize(
    "response",
    [SimpleNamespace(choices=[]), SimpleNamespace(choices=None), SimpleNamespace()],
    ids=["empty", "null-choices", "missing-attr"],
)
def test_openai_text_rejects_responses_without_choices(response: Any) -> None:
    # The old code did response.choices[0].message.content, which raised
    # IndexError on a content-filtered reply.
    with pytest.raises(ValueError, match="no choices"):
        _openai_text(response)


def test_openai_text_reports_finish_reason_on_empty_content() -> None:
    # A truncated stream yields a choice whose content is empty/None; the error
    # must mention finish_reason so the cause (e.g. "length") is diagnosable.
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="length",
            )
        ]
    )
    with pytest.raises(ValueError, match="finish_reason='length'"):
        _openai_text(response)
