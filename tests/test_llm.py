"""Contracts for :mod:`furrow.llm`: sandboxed file helpers and the tool-use loop.

Two independent halves are pinned here:

* **Path confinement.** Every helper resolves through :meth:`LLMClient.resolve_path`
  and refuses anything that lands outside ``settings.workspace`` -- ``../``,
  absolute, ``~``, and symlinks that point out. :meth:`LLMClient.write_file`
  additionally refuses a target that resolves to a *directory*; that guard is what
  stops the ``.<name>.furrow-tmp`` staging file from being created in the target's
  *parent*, i.e. outside the sandbox.
* **Provider-agnostic text and tool handling.** Both the Anthropic and the OpenAI
  response shapes must yield plain model text, feed each tool result back into the
  conversation, and never leak a pydantic repr such as ``ToolUseBlock(...)`` as if
  the model had written it.

The provider SDKs are replaced with recording doubles (no sockets, no keys); the
workspace is a real temporary directory so the sandbox logic under test is the
production logic.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

from furrow import config
from furrow.config import Provider, Settings
from furrow.llm import (
    DEFAULT_SYSTEM_PROMPT,
    IGNORED_DIRS,
    LLMClient,
)
from tests.conftest import make_settings

# --------------------------------------------------------------------------
# Helpers / doubles
# --------------------------------------------------------------------------


@pytest.fixture
def nested_workspace(tmp_path: Path) -> tuple[Settings, Path, Path]:
    """``(settings, outer, outside)`` where the workspace sits inside ``outer``.

    ``outer`` is created by this test alone, so "the sandbox leaked" can be
    asserted as an exact file set instead of a spot check.
    """
    outer = tmp_path / "outer"
    root = outer / "ws"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("# scratch\n", encoding="utf-8")
    outside = outer / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("classified\n", encoding="utf-8")
    return make_settings(workspace=root), outer, outside


def files_under(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a file from the project.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
]


# -- Anthropic-shaped doubles -------------------------------------------------


class ThinkingBlock:
    """A reasoning block: it carries ``.thinking`` and has no ``.text`` at all.

    ``getattr(block, "text", default)`` must swallow the :class:`AttributeError`
    from the property, which is what the real SDK objects do.
    """

    type = "thinking"

    def __init__(self, thinking: str = "weighing the options", **extra: Any) -> None:
        self.thinking = thinking
        for key, value in extra.items():
            setattr(self, key, value)

    @property
    def text(self) -> str:
        raise AttributeError("a thinking block has no text")


class AnthropicBlock:
    """One entry of ``Message.content``."""

    def __init__(
        self,
        type: str,
        *,
        text: str | None = None,
        id: str = "",
        name: str = "",
        input: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self.type = type
        self.text = text
        self.id = id
        self.name = name
        self.input = input if input is not None else {}
        for key, value in extra.items():
            setattr(self, key, value)

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.type,
            "text": self.text,
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }
        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}
        return data


class AnthropicResponse:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class FakeMessagesAPI:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(f"provider called {len(self.calls)} times, more than scripted")
        return self.responses.pop(0)


class FakeAnthropicProvider:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = FakeMessagesAPI(responses)


# -- OpenAI-shaped doubles -----------------------------------------------------


class OpenAIFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class OpenAIToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.type = "function"
        self.function = OpenAIFunction(name, arguments)


class OpenAIMessage:
    def __init__(self, content: Any = None, tool_calls: list[OpenAIToolCall] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class OpenAIChoice:
    def __init__(self, message: OpenAIMessage) -> None:
        self.message = message


class OpenAIResponse:
    def __init__(self, *messages: OpenAIMessage) -> None:
        self.choices = [OpenAIChoice(message) for message in messages]


class FakeCompletionsAPI:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(f"provider called {len(self.calls)} times, more than scripted")
        return self.responses.pop(0)


class FakeOpenAIProvider:
    def __init__(self, responses: list[Any]) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletionsAPI(responses)})()


def anthropic_client(workspace: Path, responses: list[Any]) -> LLMClient:
    client = LLMClient(make_settings(provider=Provider.ANTHROPIC, workspace=workspace))
    client._anthropic = FakeAnthropicProvider(responses)  # type: ignore[assignment]
    return client


def openai_client(workspace: Path, responses: list[Any]) -> LLMClient:
    client = LLMClient(make_settings(provider=Provider.OPENAI, workspace=workspace))
    client._openai = FakeOpenAIProvider(responses)  # type: ignore[assignment]
    return client


class Recorder:
    """An ``execute`` callback that records calls and returns a fixed string."""

    def __init__(self, result: str = "contents of a.txt", error: BaseException | None = None):
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, name: str, tool_input: dict[str, Any]) -> str:
        self.calls.append((name, tool_input))
        if self.error is not None:
            raise self.error
        return self.result


# --------------------------------------------------------------------------
# Settings resolution
# --------------------------------------------------------------------------


def test_a_bare_client_still_resolves_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = make_settings(workspace=Path.cwd())
    monkeypatch.setattr(config, "settings", sentinel)

    assert LLMClient().settings is sentinel
    assert LLMClient(settings=None).settings is sentinel


def test_an_explicit_settings_object_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = make_settings(workspace=Path.cwd())
    monkeypatch.setattr(config, "settings", make_settings())

    assert LLMClient(sentinel).settings is sentinel


# --------------------------------------------------------------------------
# Path sandbox
# --------------------------------------------------------------------------


def test_a_normal_relative_path_resolves_inside_the_workspace(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = LLMClient(settings)

    assert client.resolve_path("src/app.py") == outer / "ws" / "src" / "app.py"
    assert client.resolve_path("src/../README.md") == outer / "ws" / "README.md"
    assert client.resolve_path(".") == outer / "ws"


def test_an_absolute_path_inside_the_workspace_is_accepted(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = LLMClient(settings)

    assert client.resolve_path(outer / "ws" / "src" / "app.py") == (outer / "ws" / "src" / "app.py")


@pytest.mark.parametrize("escapes", ["../escaped.txt", "src/../../escaped.txt", "../outside"])
def test_a_relative_escape_is_refused(
    nested_workspace: tuple[Settings, Path, Path], escapes: str
) -> None:
    settings, _, _ = nested_workspace
    client = LLMClient(settings)

    with pytest.raises(ValueError, match="outside workspace"):
        client.resolve_path(escapes)


def test_an_absolute_path_outside_the_workspace_is_refused(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, outside = nested_workspace
    client = LLMClient(settings)

    with pytest.raises(ValueError, match="outside workspace"):
        client.resolve_path(outside / "secret.txt")
    with pytest.raises(ValueError, match="outside workspace"):
        client.resolve_path(outer)


def test_a_home_relative_path_is_refused(
    nested_workspace: tuple[Settings, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, outer, _ = nested_workspace
    monkeypatch.setenv("HOME", str(outer))
    monkeypatch.setenv("USERPROFILE", str(outer))
    client = LLMClient(settings)

    with pytest.raises(ValueError, match="outside workspace"):
        client.resolve_path("~/escaped.txt")


def test_a_symlink_pointing_out_of_the_workspace_is_refused(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, outside = nested_workspace
    link = outer / "ws" / "link.txt"
    link.symlink_to(outside / "secret.txt")
    client = LLMClient(settings)

    with pytest.raises(ValueError, match="outside workspace"):
        client.resolve_path("link.txt")
    with pytest.raises(ValueError, match="outside workspace"):
        client.resolve_path(str(link))


# --------------------------------------------------------------------------
# read_file
# --------------------------------------------------------------------------


async def test_read_file_returns_the_contents(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    (outer / "ws" / "x.txt").write_text("alpha\n", encoding="utf-8")
    client = LLMClient(settings)

    assert await client.read_file("x.txt") == "alpha\n"
    assert await client.read_file(str(outer / "ws" / "x.txt")) == "alpha\n"


async def test_read_file_refuses_a_path_outside_the_workspace(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, outside = nested_workspace
    client = LLMClient(settings)

    with pytest.raises(ValueError, match="outside workspace"):
        await client.read_file("../outside/secret.txt")
    with pytest.raises(ValueError, match="outside workspace"):
        await client.read_file(str(outside / "secret.txt"))


async def test_read_file_decodes_undecodable_bytes_without_raising(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    (outer / "ws" / "bin.dat").write_bytes(b"ok\xff\xfe")
    client = LLMClient(settings)

    assert await client.read_file("bin.dat") == "ok\ufffd\ufffd"


# --------------------------------------------------------------------------
# write_file
# --------------------------------------------------------------------------


async def test_write_file_creates_missing_parent_directories(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = LLMClient(settings)

    await client.write_file("deep/nested/dir/x.txt", "hello")

    assert (outer / "ws" / "deep" / "nested" / "dir" / "x.txt").read_text(encoding="utf-8") == (
        "hello"
    )


async def test_write_file_overwrites_existing_content(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    (outer / "ws" / "x.txt").write_text("old and long\n", encoding="utf-8")
    client = LLMClient(settings)

    await client.write_file("x.txt", "new")

    assert (outer / "ws" / "x.txt").read_text(encoding="utf-8") == "new"


async def test_write_file_leaves_no_staging_file_behind(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = LLMClient(settings)

    await client.write_file("x.txt", "hello")

    assert files_under(outer) == {"ws/README.md", "ws/x.txt", "outside/secret.txt"}


@pytest.mark.parametrize(
    ("target", "reason"),
    [(".", "directory"), ("src", "directory"), ("../outside", "outside workspace")],
)
async def test_write_file_refuses_a_directory_target_without_stranding_a_temp_file(
    nested_workspace: tuple[Settings, Path, Path], tmp_path: Path, target: str, reason: str
) -> None:
    """Regression: the ``.<name>.furrow-tmp`` staging file escaped the sandbox.

    ``target.with_name(...)`` puts the staging file in ``target.parent``, so a
    directory target such as ``"."`` wrote ``outer/.ws.furrow-tmp`` -- *outside*
    ``settings.workspace`` -- and then failed on ``os.replace``.
    """
    settings, outer, _ = nested_workspace
    client = LLMClient(settings)

    with pytest.raises(ValueError, match=reason):
        await client.write_file(target, "pwned")

    # The exact location the unfixed code used, and the whole private tree.
    assert not (outer / ".ws.furrow-tmp").exists()
    assert list(tmp_path.rglob("*.furrow-tmp")) == []
    assert files_under(outer) == {"ws/README.md", "outside/secret.txt"}


async def test_write_file_refuses_a_path_outside_the_workspace(
    nested_workspace: tuple[Settings, Path, Path], tmp_path: Path
) -> None:
    settings, outer, outside = nested_workspace
    client = LLMClient(settings)

    with pytest.raises(ValueError, match="outside workspace"):
        await client.write_file("../escaped.txt", "pwned")
    with pytest.raises(ValueError, match="outside workspace"):
        await client.write_file(str(outside / "escaped.txt"), "pwned")

    assert not (outer / "escaped.txt").exists()
    assert not (outside / "escaped.txt").exists()
    assert list(tmp_path.rglob("*.furrow-tmp")) == []


# --------------------------------------------------------------------------
# list_files (synchronous)
# --------------------------------------------------------------------------


async def test_list_files_is_synchronous(nested_workspace: tuple[Settings, Path, Path]) -> None:
    settings, _, _ = nested_workspace
    client = LLMClient(settings)

    result = client.list_files()

    assert not inspect.isawaitable(result)
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


async def test_list_files_skips_noise_directories_and_sorts_workspace_relative_paths(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    root = outer / "ws"
    for name in (".git", "__pycache__", "node_modules", ".venv", ".pytest_cache", ".ruff_cache"):
        (root / name).mkdir()
        (root / name / "noise.pyc").write_text("noise\n", encoding="utf-8")
    (root / "node_modules" / "deep").mkdir()
    (root / "node_modules" / "deep" / "deeper.py").write_text("noise\n", encoding="utf-8")
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    (root / "src" / "b.py").write_text("b\n", encoding="utf-8")
    (root / "src" / "deep").mkdir()
    (root / "src" / "deep" / "c.py").write_text("c\n", encoding="utf-8")
    client = LLMClient(settings)

    found = client.list_files()

    assert found == ["README.md", "a.txt", "src/b.py", "src/deep/c.py"]
    assert found == sorted(found)
    assert all(not Path(item).is_absolute() for item in found)
    for skipped in IGNORED_DIRS:
        assert not any(part == skipped for item in found for part in Path(item).parts)


async def test_list_files_can_be_scoped_to_a_subdirectory(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    (outer / "ws" / "src" / "b.py").write_text("b\n", encoding="utf-8")
    client = LLMClient(settings)

    assert client.list_files("src") == ["b.py"]


async def test_list_files_returns_empty_for_a_file_or_a_missing_directory(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    (outer / "ws" / "x.txt").write_text("x\n", encoding="utf-8")
    client = LLMClient(settings)

    assert client.list_files("x.txt") == []
    assert client.list_files("nope") == []


async def test_list_files_refuses_a_directory_outside_the_workspace(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, _, _ = nested_workspace
    client = LLMClient(settings)

    with pytest.raises(ValueError, match="outside workspace"):
        client.list_files("..")


# --------------------------------------------------------------------------
# agent_loop -- Anthropic shape
# --------------------------------------------------------------------------


async def test_anthropic_loop_runs_the_tool_feeds_back_the_result_and_returns_the_text(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = anthropic_client(
        outer / "ws",
        [
            AnthropicResponse(
                [AnthropicBlock("tool_use", id="tu_1", name="read_file", input={"path": "a.txt"})]
            ),
            AnthropicResponse([AnthropicBlock("text", text="read a.txt")]),
        ],
    )
    execute = Recorder("contents of a.txt")

    result = await client.agent_loop("read a.txt", TOOLS, execute, system="be helpful")

    assert result == "read a.txt"
    assert execute.calls == [("read_file", {"path": "a.txt"})]

    first, second = client._anthropic.messages.calls
    assert first["tools"] == TOOLS
    assert first["system"] == "be helpful"
    assert first["model"] == client.settings.model
    # The client reuses one mutable ``messages`` list across provider calls, so
    # the dict recorded on the first call aliases the live list and has since
    # grown -- assert the full final conversation on the second call instead.
    assert second["messages"] == [
        {"role": "user", "content": "read a.txt"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "read_file",
                    "input": {"path": "a.txt"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_1", "content": "contents of a.txt"}
            ],
        },
    ]
    assert "is_error" not in second["messages"][-1]["content"][0]
    assert second["messages"][-2]["role"] == "assistant"


async def test_anthropic_loop_defaults_the_system_prompt_when_none_is_given(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = anthropic_client(
        outer / "ws", [AnthropicResponse([AnthropicBlock("text", text="ok")])]
    )

    assert await client.agent_loop("hi", [], Recorder()) == "ok"
    assert client._anthropic.messages.calls[0]["system"] == DEFAULT_SYSTEM_PROMPT
    assert "tools" not in client._anthropic.messages.calls[0]


async def test_anthropic_loop_with_max_turns_one_stops_after_a_single_provider_call(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = anthropic_client(
        outer / "ws",
        [
            AnthropicResponse(
                [
                    AnthropicBlock("text", text="working on it"),
                    AnthropicBlock(
                        "tool_use", id="tu_1", name="read_file", input={"path": "a.txt"}
                    ),
                ]
            )
        ],
    )
    execute = Recorder()

    result = await client.agent_loop("read a.txt", TOOLS, execute, max_turns=1)

    assert result == "working on it"
    assert len(client._anthropic.messages.calls) == 1
    assert execute.calls == [("read_file", {"path": "a.txt"})]


async def test_a_raising_execute_becomes_a_tool_error_and_the_loop_continues(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = anthropic_client(
        outer / "ws",
        [
            AnthropicResponse(
                [AnthropicBlock("tool_use", id="tu_1", name="read_file", input={"path": "a.txt"})]
            ),
            AnthropicResponse([AnthropicBlock("text", text="recovered")]),
        ],
    )
    execute = Recorder(error=RuntimeError("disk on fire"))

    result = await client.agent_loop("read a.txt", TOOLS, execute)

    assert result == "recovered"
    fed_back = client._anthropic.messages.calls[1]["messages"][-1]["content"][0]
    assert fed_back == {
        "type": "tool_result",
        "tool_use_id": "tu_1",
        "content": "Error: RuntimeError: disk on fire",
        "is_error": True,
    }


async def test_anthropic_text_extraction_skips_a_thinking_block(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = anthropic_client(
        outer / "ws",
        [
            AnthropicResponse([ThinkingBlock(), AnthropicBlock("text", text="the answer")]),
        ],
    )

    assert await client.complete("think then answer") == "the answer"


async def test_anthropic_text_extraction_of_a_tool_only_turn_is_empty_not_a_repr(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    """Regression: a tool-only turn used to leak ``ToolUseBlock(...)`` as text."""
    settings, outer, _ = nested_workspace
    client = anthropic_client(
        outer / "ws",
        [
            AnthropicResponse(
                [
                    AnthropicBlock(
                        "tool_use", id="tu_1", name="read_file", input={"path": "a.txt"}
                    ),
                    ThinkingBlock(),
                ]
            )
        ],
    )

    text = await client.complete("do the thing")

    assert text == ""
    assert "ToolUseBlock(" not in text
    assert "model_dump" not in text


async def test_the_loop_returns_the_last_real_text_when_turns_run_out(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    """The second turn carries no text; the first turn's answer must survive."""
    settings, outer, _ = nested_workspace
    client = anthropic_client(
        outer / "ws",
        [
            AnthropicResponse(
                [
                    AnthropicBlock("text", text="partial answer"),
                    AnthropicBlock(
                        "tool_use", id="tu_1", name="read_file", input={"path": "a.txt"}
                    ),
                ]
            ),
            AnthropicResponse(
                [
                    AnthropicBlock(
                        "tool_use", id="tu_2", name="read_file", input={"path": "b.txt"}
                    ),
                    ThinkingBlock(),
                ]
            ),
        ],
    )
    execute = Recorder()

    result = await client.agent_loop("read both", TOOLS, execute, max_turns=2)

    assert result == "partial answer"
    assert "ToolUseBlock(" not in result
    assert "model_dump" not in result
    assert [name for name, _ in execute.calls] == ["read_file", "read_file"]


async def test_anthropic_complete_uses_the_configured_model(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = anthropic_client(outer / "ws", [AnthropicResponse([AnthropicBlock("text", text="x")])])

    await client.complete("hi", system="sys", model="override-model")

    call = client._anthropic.messages.calls[0]
    assert call["model"] == "override-model"
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------
# agent_loop -- OpenAI shape
# --------------------------------------------------------------------------


async def test_openai_loop_runs_the_tool_feeds_back_the_result_and_returns_the_text(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = openai_client(
        outer / "ws",
        [
            OpenAIResponse(
                OpenAIMessage(
                    tool_calls=[
                        OpenAIToolCall("call_1", "read_file", '{"path": "a.txt", "limit": 10}')
                    ]
                )
            ),
            OpenAIResponse(OpenAIMessage("read a.txt")),
        ],
    )
    execute = Recorder("contents of a.txt")

    result = await client.agent_loop("read a.txt", TOOLS, execute, system="be helpful")

    assert result == "read a.txt"
    assert execute.calls == [("read_file", {"path": "a.txt", "limit": 10})]

    first, second = client._openai.chat.completions.calls
    # Anthropic-style schemas are translated for OpenAI by the client itself.
    assert first["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the project.",
                "parameters": TOOLS[0]["input_schema"],
            },
        }
    ]
    assert first["tool_choice"] == "auto"
    # The client reuses one mutable ``messages`` list across provider calls, so
    # the dict recorded on the first call aliases the live list and has since
    # grown -- assert the full final conversation on the second call instead.
    assert second["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "read a.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "a.txt", "limit": 10}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "contents of a.txt",
        },
    ]
    assert second["messages"][-1]["role"] == "tool"
    assert second["messages"][-2]["role"] == "assistant"


async def test_openai_loop_with_max_turns_one_stops_after_a_single_provider_call(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = openai_client(
        outer / "ws",
        [
            OpenAIResponse(
                OpenAIMessage(
                    tool_calls=[OpenAIToolCall("call_1", "read_file", '{"path": "a.txt"}')]
                )
            )
        ],
    )
    execute = Recorder()

    result = await client.agent_loop("read a.txt", TOOLS, execute, max_turns=1)

    assert result == ""
    assert len(client._openai.chat.completions.calls) == 1
    assert execute.calls == [("read_file", {"path": "a.txt"})]


async def test_a_raising_execute_becomes_a_tool_error_and_the_openai_loop_continues(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = openai_client(
        outer / "ws",
        [
            OpenAIResponse(
                OpenAIMessage(
                    tool_calls=[OpenAIToolCall("call_1", "read_file", '{"path": "a.txt"}')]
                )
            ),
            OpenAIResponse(OpenAIMessage("recovered")),
        ],
    )
    execute = Recorder(error=RuntimeError("disk on fire"))

    result = await client.agent_loop("read a.txt", TOOLS, execute)

    assert result == "recovered"
    assert client._openai.chat.completions.calls[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Error: RuntimeError: disk on fire",
    }


async def test_openai_loop_returns_the_last_real_text_when_turns_run_out(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = openai_client(
        outer / "ws",
        [
            OpenAIResponse(
                OpenAIMessage(
                    content="partial answer",
                    tool_calls=[OpenAIToolCall("call_1", "read_file", '{"path": "a.txt"}')],
                )
            ),
            OpenAIResponse(
                OpenAIMessage(
                    tool_calls=[OpenAIToolCall("call_2", "read_file", '{"path": "b.txt"}')]
                )
            ),
        ],
    )
    execute = Recorder()

    result = await client.agent_loop("read both", TOOLS, execute, max_turns=2)

    assert result == "partial answer"
    assert [name for name, _ in execute.calls] == ["read_file", "read_file"]


async def test_openai_unparseable_tool_arguments_become_an_empty_payload(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = openai_client(
        outer / "ws",
        [
            OpenAIResponse(
                OpenAIMessage(tool_calls=[OpenAIToolCall("call_1", "read_file", "not json")])
            ),
            OpenAIResponse(OpenAIMessage("ok")),
        ],
    )
    execute = Recorder()

    result = await client.agent_loop("read a.txt", TOOLS, execute)

    assert result == "ok"
    assert execute.calls == [("read_file", {})]


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------


async def test_openai_complete_joins_text_content_parts(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = openai_client(
        outer / "ws",
        [
            OpenAIResponse(
                OpenAIMessage(
                    content=[
                        {"type": "reasoning", "reasoning": "hmm"},
                        {"type": "text", "text": "part one"},
                        {"type": "text", "text": "part two"},
                    ]
                )
            )
        ],
    )

    assert await client.complete("hi", system="sys") == "part one\npart two"
    call = client._openai.chat.completions.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


async def test_openai_complete_tolerates_null_content_and_missing_choices(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = openai_client(outer / "ws", [OpenAIResponse(OpenAIMessage(content=None))])

    assert await client.complete("hi") == ""

    empty = openai_client(outer / "ws", [type("Empty", (), {"choices": []})()])
    assert await empty.complete("hi") == ""


async def test_an_unsupported_provider_is_refused_clearly(
    nested_workspace: tuple[Settings, Path, Path],
) -> None:
    settings, outer, _ = nested_workspace
    client = LLMClient(make_settings(provider="ollama", workspace=outer / "ws"))

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hi")
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.agent_loop("hi", TOOLS, Recorder())
