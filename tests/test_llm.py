"""Tests for :mod:`furrow.llm`: settings resolution, JSON recovery and workspace listing.

One section per defect guarded here, and every test in the file is offline: no provider is
contacted, no API key is required and nothing is asserted about agent *behaviour* (that
belongs to ``tests/test_agents.py``).

* :class:`~furrow.llm.LLMClient` used to store the ``None`` it was handed, so the first
  attribute read on ``client.settings`` raised ``AttributeError: 'NoneType' object has no
  attribute ...``. The three agents are constructed with no arguments on the production
  path, so that was a crash rather than a cosmetic issue.
* :func:`~furrow.llm.extract_json` did not exist; raw replies went through a bare
  ``json.loads`` that fails on a fenced or chatty reply.
* :meth:`~furrow.llm.LLMClient.list_files` used ``Path.rglob``, which descends into
  ``.git`` and ``node_modules`` and pushed thousands of irrelevant paths into prompts.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

# ``TesterAgent`` is reached through the module, not imported by name: pytest tries to
# collect any module level name starting with ``Test`` and would warn about a class that
# has a constructor. See the same class of problem in ``tests/conftest.py``.
from furrow.agents import tester as tester_module
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Provider, Settings, TaskModel
from furrow.config import settings as module_settings
from furrow.llm import IGNORED_DIRECTORIES, LLMClient, extract_json

# --------------------------------------------------------------------------------------
# settings resolution
# --------------------------------------------------------------------------------------


def _build_worker() -> WorkerAgent:
    """A worker with the one argument its signature requires and nothing else."""
    return WorkerAgent(task=TaskModel(id="t1", description="write a module"))


#: ``(case name, factory)`` for the three agents, built exactly as the orchestrator builds
#: them when it has no scripted client to inject: no ``client`` and no ``settings``.
_NO_ARGUMENT_AGENTS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("PlannerAgent", PlannerAgent),
    ("WorkerAgent", _build_worker),
    ("TesterAgent", tester_module.TesterAgent),
)

_NO_ARGUMENT_AGENT_IDS = [name for name, _ in _NO_ARGUMENT_AGENTS]
_NO_ARGUMENT_AGENT_FACTORIES = [build for _, build in _NO_ARGUMENT_AGENTS]


def test_client_settings_none_resolves_to_usable_settings() -> None:
    """``LLMClient(settings=None)`` used to leave ``self.settings`` as ``None``."""
    client = LLMClient(settings=None)

    assert client.settings is not None
    assert isinstance(client.settings, Settings)
    # The two fields ``complete()`` reads first: reaching them is the actual crash path.
    assert isinstance(client.settings.model, str)
    assert client.settings.model
    assert isinstance(client.settings.provider, Provider)


def test_client_without_arguments_uses_the_module_level_settings() -> None:
    """The no-argument default and an explicit ``None`` mean the same thing."""
    assert LLMClient().settings is module_settings
    assert LLMClient(settings=None).settings is module_settings


def test_client_honours_an_explicitly_passed_settings_instance() -> None:
    """A caller that supplies its own settings keeps exactly that object."""
    supplied = Settings(model="a-model-only-this-test-uses")

    assert LLMClient(settings=supplied).settings is supplied
    assert LLMClient(settings=supplied).settings.model == "a-model-only-this-test-uses"


@pytest.mark.parametrize("build", _NO_ARGUMENT_AGENT_FACTORIES, ids=_NO_ARGUMENT_AGENT_IDS)
def test_agents_built_without_arguments_resolve_usable_settings(
    build: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every agent's own settings attribute resolves, and needs no API key to exist."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    agent = build()

    assert isinstance(agent.client.settings, Settings)
    assert agent.client.settings is module_settings
    assert agent.settings is agent.client.settings
    assert agent.settings.model
    assert isinstance(agent.settings.provider, Provider)
    # Construction must not have reached for a provider: with both keys removed, the lazy
    # property is the thing that has to complain, not the constructor.
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        _ = agent.client.anthropic


# --------------------------------------------------------------------------------------
# extract_json
# --------------------------------------------------------------------------------------

#: ``(case name, raw model reply, expected decoded value)``.
_RECOVERABLE_REPLIES: tuple[tuple[str, str, Any], ...] = (
    ("bare_object", '{"a": 1, "b": [2, 3]}', {"a": 1, "b": [2, 3]}),
    ("json_fenced_block", '```json\n{"a": 1}\n```', {"a": 1}),
    ("plain_fenced_block", "```\n[1, 2, 3]\n```", [1, 2, 3]),
    (
        "fenced_block_with_trailing_prose",
        '```json\n{"a": 1}\n```\nThat is the plan.',
        {"a": 1},
    ),
    (
        "wrapped_in_surrounding_prose",
        'Sure!\nHere is the plan:\n{"a": 1}\nLet me know if that works.',
        {"a": 1},
    ),
    (
        "deeply_nested_objects_and_arrays",
        '{"a": {"b": {"c": {"d": [1, {"e": [2, 3]}], "f": []}}}}',
        {"a": {"b": {"c": {"d": [1, {"e": [2, 3]}], "f": []}}}},
    ),
    ("closing_brace_inside_a_string_value", r'{"note": "a } b"}', {"note": "a } b"}),
    ("escaped_quote_inside_a_string", r'{"note": "say \"hi\""}', {"note": 'say "hi"'}),
    ("escaped_backslash_then_quote", r'{"note": "a \\\" b"}', {"note": 'a \\" b'}),
    ("escaped_backslash_in_a_string", r'{"note": "back\\slash"}', {"note": "back\\slash"}),
    ("top_level_array", '[{"id": "1"}, {"id": "2"}]', [{"id": "1"}, {"id": "2"}]),
    ("leading_and_trailing_whitespace", '\n\n\t  {"a": 1}  \n\n', {"a": 1}),
)


_RECOVER_IDS = [case for case, _, _ in _RECOVERABLE_REPLIES]
_RECOVER_ARGS = [(reply, expected) for _, reply, expected in _RECOVERABLE_REPLIES]


@pytest.mark.parametrize(("reply", "expected"), _RECOVER_ARGS, ids=_RECOVER_IDS)
def test_extract_json_recovers_the_payload_from_a_model_reply(reply: str, expected: Any) -> None:
    assert extract_json(reply) == expected


def test_extract_json_finds_the_object_behind_a_leading_json_shaped_brace() -> None:
    """A ``}`` inside a string must not end the scan early."""
    reply = 'Here you go: {"note": "a } b", "count": 2} -- enjoy!'

    assert extract_json(reply) == {"note": "a } b", "count": 2}


def test_extract_json_agrees_with_json_loads_on_every_recoverable_case() -> None:
    """Cross-check: whatever ``extract_json`` returns must be genuine decoded JSON."""
    for _case, reply, expected in _RECOVERABLE_REPLIES:
        assert json.dumps(extract_json(reply), sort_keys=True) == json.dumps(
            expected, sort_keys=True
        )


_UNPARSEABLE_REPLIES: tuple[tuple[str, Any], ...] = (
    ("empty_string", ""),
    ("whitespace_only", "   \n\t "),
    ("prose_only", "Sure! I could not find anything structured to report."),
    ("html_error_page", "<html><body>502 Bad Gateway</body></html>"),
    ("python_repr_not_json", "{'single': 'quotes'}"),
    ("truncated_object", '{"unterminated": '),
    ("truncated_array", "[1, 2"),
    ("fenced_block_without_json", "```\nno json here\n```"),
    ("unbalanced_closer", "} stray }"),
    ("none_reply", None),
    ("integer_reply", 42),
    ("bytes_reply", b'{"a": 1}'),
    ("object_reply", {"a": 1}),
)


@pytest.mark.parametrize(
    "reply",
    [reply for _, reply in _UNPARSEABLE_REPLIES],
    ids=[case for case, _ in _UNPARSEABLE_REPLIES],
)
def test_extract_json_raises_value_error_for_unusable_replies(reply: Any) -> None:
    """Anything that carries no JSON value is a decode failure, never a crash."""
    with pytest.raises(ValueError) as excinfo:
        extract_json(reply)

    assert str(excinfo.value)


def test_extract_json_failure_message_previews_a_truncated_reply() -> None:
    """A huge unusable reply must not be echoed whole into the error message."""
    garbage = "this is not json " * 2000
    assert len(garbage) > 10_000

    with pytest.raises(ValueError) as excinfo:
        extract_json(garbage)

    message = str(excinfo.value)
    assert len(message) < 500
    assert len(message) < len(garbage) // 10
    assert "..." in message
    assert "this is not json" in message
    assert garbage not in message


def test_extract_json_failure_message_is_a_single_line() -> None:
    """A multi-line reply is collapsed, so the message stays loggable."""
    with pytest.raises(ValueError) as excinfo:
        extract_json("line one\nline two\n\tline three")

    message = str(excinfo.value)
    assert "\n" not in message
    assert message.endswith("'")


# --------------------------------------------------------------------------------------
# list_files
# --------------------------------------------------------------------------------------


@pytest.fixture
def client() -> LLMClient:
    """A client on the default settings. Nothing below ever reaches a provider."""
    return LLMClient(settings=None)


#: Every name the module claims to prune, read from the source so the test cannot drift
#: away from a list that is edited later.
IGNORED_DIRECTORY_NAMES: tuple[str, ...] = tuple(sorted(IGNORED_DIRECTORIES))

#: The noise a real checkout is full of, named in the brief.
USUAL_NOISE_DIRECTORIES: tuple[str, ...] = (
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)

#: A plausible project layout, kept sorted so the expected listing is a plain ``list``.
PROJECT_FILES: tuple[str, ...] = (
    "README.md",
    "furrow/agents/planner.py",
    "furrow/llm.py",
    "pyproject.toml",
)


def plant_noise(root: Path, directories: Iterable[str]) -> None:
    """Create each named directory under ``root``, holding files that must never surface."""
    for name in directories:
        inside = root / name
        (inside / "nested").mkdir(parents=True)
        (inside / "payload.txt").write_text("noise\n", encoding="utf-8")
        (inside / "nested" / "deep.txt").write_text("noise\n", encoding="utf-8")


def plant_project(root: Path) -> None:
    """Create the :data:`PROJECT_FILES` layout under ``root``."""
    for relative in PROJECT_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# placeholder\n", encoding="utf-8")


@pytest.mark.parametrize("ignored", IGNORED_DIRECTORY_NAMES)
def test_list_files_prunes_each_ignored_directory(
    client: LLMClient, tmp_path: Path, ignored: str
) -> None:
    """One case per member of ``IGNORED_DIRECTORIES``, so a new name is covered for free."""
    plant_noise(tmp_path, [ignored])
    (tmp_path / "kept.txt").write_text("keep\n", encoding="utf-8")

    assert client.list_files(tmp_path) == ["kept.txt"]


def test_list_files_prunes_the_usual_noise_directories(
    client: LLMClient, tmp_path: Path
) -> None:
    plant_noise(tmp_path, USUAL_NOISE_DIRECTORIES)
    plant_project(tmp_path)

    result = client.list_files(tmp_path)

    assert result == list(PROJECT_FILES)
    leaked = [path for path in result if set(Path(path).parts) & set(IGNORED_DIRECTORIES)]
    assert leaked == []


def test_list_files_returns_paths_relative_to_the_given_root(
    client: LLMClient, tmp_path: Path
) -> None:
    plant_noise(tmp_path, [".git"])
    plant_project(tmp_path)
    root = tmp_path.resolve()

    result = client.list_files(root)

    assert result
    for path in result:
        candidate = Path(path)
        assert not candidate.is_absolute()
        assert ".." not in candidate.parts
        assert (root / candidate).is_file()


def test_list_files_result_is_sorted(client: LLMClient, tmp_path: Path) -> None:
    plant_project(tmp_path)
    (tmp_path / "zzz.txt").write_text("z\n", encoding="utf-8")
    (tmp_path / "aaa.txt").write_text("a\n", encoding="utf-8")

    result = client.list_files(tmp_path)

    assert result == sorted(result)
    # A plain code-point sort, not a case-folded one: the assertions would not care.
    assert result[0] == "README.md"


def test_list_files_result_is_deduplicated_and_stable_across_calls(
    client: LLMClient, tmp_path: Path
) -> None:
    plant_noise(tmp_path, ["node_modules", "__pycache__", "dist"])
    plant_project(tmp_path)

    first = client.list_files(tmp_path)
    second = client.list_files(tmp_path)

    assert first == list(PROJECT_FILES)
    assert len(first) == len(set(first))
    # A second call must not double anything, whatever state the walk keeps.
    assert second == first


def test_list_files_returns_empty_list_for_a_missing_directory(
    client: LLMClient, tmp_path: Path
) -> None:
    assert client.list_files(tmp_path / "does-not-exist") == []


def test_list_files_returns_empty_list_when_the_root_is_a_file(
    client: LLMClient, tmp_path: Path
) -> None:
    target = tmp_path / "solo.txt"
    target.write_text("x\n", encoding="utf-8")

    assert client.list_files(target) == []


def test_list_files_returns_empty_list_for_an_empty_directory(
    client: LLMClient, tmp_path: Path
) -> None:
    (tmp_path / "empty").mkdir()

    assert client.list_files(tmp_path / "empty") == []


def test_list_files_accepts_a_relative_root(
    client: LLMClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planner hands over whatever path the settings hold, which may be relative."""
    plant_noise(tmp_path, [".git"])
    plant_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert client.list_files(".") == list(PROJECT_FILES)
    assert client.list_files("furrow") == ["agents/planner.py", "llm.py"]
