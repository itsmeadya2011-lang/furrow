"""Contracts for :mod:`furrow.agents.worker`.

Three things matter here and are pinned below:

* the **tool loop really edits the workspace** -- a scripted ``write_file`` then
  ``edit_file`` must land on disk;
* the worker **confines every path** to ``settings.workspace``, whether the file
  helpers come from the injected client or from the agent's own fallbacks, and
  reports a refusal as text the model can read rather than an exception;
* :meth:`WorkerAgent.run` **never raises** -- the orchestrator schedules the next
  cycle from the results of every task, so an escaping exception would abort the
  whole cycle.

The harness in :mod:`tests.conftest` supplies the scripts (``FakeLLM``,
``MinimalClient``) and delegates the sandboxed file helpers to a real
:class:`~furrow.llm.LLMClient`, so path confinement under test is the production
behaviour and not a re-implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.worker import WORKER_TOOLS, WorkerAgent, WorkerResult
from furrow.config import Settings, TaskModel
from tests.conftest import FakeLLM, MinimalClient, ToolScript, make_settings

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


class RecordingScript:
    """A :class:`ToolScript` that also keeps the text each tool returned.

    ``outputs`` is what the *model* sees, so it is what the "returns an error
    string, not an exception" contracts are asserted against.
    """

    def __init__(self, *calls: tuple[str, dict[str, Any]], summary: str = "") -> None:
        self.calls = list(calls)
        self.summary = summary
        self.outputs: list[tuple[str, str]] = []

    async def __call__(self, execute: Any) -> str:
        for name, tool_input in self.calls:
            self.outputs.append((name, str(await execute(name, tool_input))))
        return self.summary

    def output_for(self, name: str) -> str:
        """The text the model was given for the last call to tool *name*."""
        for tool_name, text in reversed(self.outputs):
            if tool_name == name:
                return text
        raise AssertionError(f"tool {name!r} was never called (saw {self.outputs!r})")


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Settings, Path]:
    """A workspace nested one level down inside an ``outer`` dir we own entirely.

    Having a private parent means "nothing escaped the workspace" can be asserted
    as an exact file set rather than a spot check.
    """
    outer = tmp_path / "outer"
    root = outer / "ws"
    (root / "src").mkdir(parents=True)
    (root / "README.md").write_text("# scratch\n", encoding="utf-8")
    return make_settings(workspace=root), outer


def files_under(root: Path) -> set[str]:
    """Every file below *root*, as sorted workspace-relative POSIX paths."""
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def make_task(description: str = "add a greeting module") -> TaskModel:
    return TaskModel(id="1", description=description)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_construction_without_a_client_or_settings_does_not_raise() -> None:
    """Regression: ``WorkerAgent(task=t)`` used to leave ``self.settings`` as None.

    ``run()`` then raised ``AttributeError: 'NoneType' object has no attribute
    'worker_model'`` before it could even reach the model.
    """
    agent = WorkerAgent(task=make_task())

    assert isinstance(agent.client.settings, Settings)
    assert isinstance(agent.settings, Settings)
    assert agent.settings.worker_model


def test_an_explicit_client_takes_precedence_over_the_settings_argument() -> None:
    settings = make_settings(workspace=Path("/tmp/never-used"))
    client = FakeLLM(settings=make_settings(workspace=Path.cwd()))

    agent = WorkerAgent(task=make_task(), client=client, settings=settings)

    assert agent.client is client
    assert agent.settings is client.settings


def test_an_explicit_settings_argument_is_used_when_the_client_has_none() -> None:
    """A client with no ``settings`` of its own must not win over the argument."""

    class SettingslessClient:
        settings: Settings | None = None

        async def agent_loop(self, *args: Any, **kwargs: Any) -> str:
            raise AssertionError("not used")

    settings = make_settings(workspace=Path.cwd())

    agent = WorkerAgent(
        task=make_task(),
        client=SettingslessClient(),
        settings=settings,  # type: ignore[arg-type]
    )

    assert agent.settings is settings


# --------------------------------------------------------------------------
# The happy path: a scripted tool loop really edits the workspace
# --------------------------------------------------------------------------


async def test_scripted_write_then_edit_lands_on_disk(workspace: tuple[Settings, Path]) -> None:
    settings, _ = workspace
    client = FakeLLM(
        settings=settings,
        scripts=[
            ToolScript(
                ("write_file", {"path": "x.txt", "content": "alpha\nbeta\n"}),
                ("edit_file", {"path": "x.txt", "old_string": "beta", "new_string": "gamma"}),
                summary="wrote x.txt",
            )
        ],
    )
    task = make_task()

    result = await WorkerAgent(task=task, client=client).run()

    assert isinstance(result, WorkerResult)
    assert result.success is True
    assert result.error is None
    assert result.summary == "wrote x.txt"
    assert result.files_changed == ["x.txt"]
    assert task.status == "completed"
    assert task.status_message is None
    assert task.result == "wrote x.txt"
    assert (settings.workspace / "x.txt").read_text(encoding="utf-8") == "alpha\ngamma\n"


async def test_the_same_loop_works_with_a_client_that_has_no_file_helpers(
    workspace: tuple[Settings, Path],
) -> None:
    """``MinimalClient`` exposes only ``settings``/``agent_loop``.

    The worker must then use its own fallbacks and still confine every path --
    these are the same outcomes as with the full client, which is the point.
    """
    settings, _ = workspace
    client = MinimalClient(
        settings=settings,
        scripts=[
            ToolScript(
                ("write_file", {"path": "x.txt", "content": "alpha\nbeta\n"}),
                ("edit_file", {"path": "x.txt", "old_string": "beta", "new_string": "gamma"}),
                summary="wrote x.txt",
            )
        ],
    )
    task = make_task()

    result = await WorkerAgent(task=task, client=client).run()

    assert result.success is True
    assert result.files_changed == ["x.txt"]
    assert task.status == "completed"
    assert task.result == "wrote x.txt"
    assert (settings.workspace / "x.txt").read_text(encoding="utf-8") == "alpha\ngamma\n"


async def test_run_passes_the_worker_prompt_tools_and_execute_callback(
    workspace: tuple[Settings, Path],
) -> None:
    settings, _ = workspace
    client = FakeLLM(settings=settings, scripts=["done"])
    agent = WorkerAgent(task=make_task("write the module"), client=client)

    await agent.run()

    call = client.agent_loop_calls[0]
    assert client.agent_loop_call_count == 1
    assert [tool["name"] for tool in call["tools"]] == [
        "read_file",
        "write_file",
        "edit_file",
        "list_files",
    ]
    assert call["system"] == WORKER_PROMPT
    assert call["model"] == settings.worker_model
    # The model may only act through the worker's dispatcher, which never raises.
    assert call["execute"].__self__ is agent
    assert "write the module" in call["prompt"]


# --------------------------------------------------------------------------
# run() never raises
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "boom",
    [RuntimeError("provider exploded"), TimeoutError("no route to host"), ValueError("bad key")],
)
async def test_a_raising_agent_loop_becomes_a_failed_result(
    workspace: tuple[Settings, Path], boom: BaseException
) -> None:
    settings, _ = workspace
    task = make_task()

    result = await WorkerAgent(task=task, client=FakeLLM(settings=settings, fail_with=boom)).run()

    assert result.success is False
    assert result.error is not None
    assert f"{type(boom).__name__}: {boom}" in result.error
    assert task.status == "failed"
    assert task.status_message == result.error


async def test_a_missing_script_is_swallowed_rather_than_propagated(
    workspace: tuple[Settings, Path],
) -> None:
    """``FakeLLM`` asserts when a scripted turn is missing; the worker must absorb it."""
    settings, _ = workspace

    result = await WorkerAgent(task=make_task(), client=FakeLLM(settings=settings)).run()

    assert result.success is False
    assert "AssertionError" in (result.error or "")


async def test_an_empty_summary_is_a_clean_result_not_a_crash(
    workspace: tuple[Settings, Path],
) -> None:
    settings, _ = workspace
    task = make_task()

    result = await WorkerAgent(task=task, client=FakeLLM(settings=settings, scripts=[""])).run()

    assert isinstance(result, WorkerResult)
    assert result.success is False
    assert result.error
    assert "summary" in result.error
    assert result.files_changed == []
    assert task.status == "failed"
    assert task.result is None


# --------------------------------------------------------------------------
# Refusal detection, including the false-positive guard
# --------------------------------------------------------------------------


async def test_a_refusal_summary_fails_the_task(workspace: tuple[Settings, Path]) -> None:
    settings, _ = workspace
    refusal = "I cannot complete this task."
    task = make_task()

    result = await WorkerAgent(
        task=task, client=FakeLLM(settings=settings, scripts=[refusal])
    ).run()

    assert result.success is False
    assert result.summary == refusal
    assert "could not complete the task" in (result.error or "")
    assert task.status == "failed"
    assert task.result is None


async def test_an_ordinary_summary_succeeds(workspace: tuple[Settings, Path]) -> None:
    settings, _ = workspace
    task = make_task()

    result = await WorkerAgent(
        task=task, client=FakeLLM(settings=settings, scripts=["wrote x.txt"])
    ).run()

    assert result.success is True
    assert result.error is None
    assert task.status == "completed"


async def test_a_summary_merely_mentioning_cannot_is_not_a_refusal(
    workspace: tuple[Settings, Path],
) -> None:
    """The refusal matcher runs on the whole summary, so it must not fire here.

    ``"could not handle"`` is not one of the phrasings that mean "I gave up";
    a real fix report has to stay ``success is True``.
    """
    settings, _ = workspace
    task = make_task()
    summary = "fixed the bug where the parser could not handle empty input"

    result = await WorkerAgent(
        task=task, client=FakeLLM(settings=settings, scripts=[summary])
    ).run()

    assert result.success is True
    assert result.summary == summary
    assert result.error is None
    assert task.status == "completed"


# --------------------------------------------------------------------------
# Path confinement
# --------------------------------------------------------------------------


async def test_a_relative_escape_is_refused_and_writes_nothing(
    workspace: tuple[Settings, Path],
) -> None:
    settings, outer = workspace
    script = RecordingScript(("write_file", {"path": "../escaped.txt", "content": "pwned"}))

    result = await WorkerAgent(
        task=make_task(), client=FakeLLM(settings=settings, scripts=[script])
    ).run()

    assert "Error:" in script.output_for("write_file")
    assert "outside workspace" in script.output_for("write_file")
    assert result.files_changed == []
    assert files_under(outer) == {"ws/README.md"}


async def test_an_absolute_path_outside_the_workspace_is_refused(
    workspace: tuple[Settings, Path],
) -> None:
    settings, outer = workspace
    outside = outer / "outside"
    outside.mkdir()
    target = outside / "escaped.txt"
    script = RecordingScript(("write_file", {"path": str(target), "content": "pwned"}))

    await WorkerAgent(task=make_task(), client=FakeLLM(settings=settings, scripts=[script])).run()

    assert "Error:" in script.output_for("write_file")
    assert not target.exists()
    assert files_under(outer) == {"ws/README.md"}


async def test_a_home_relative_path_is_refused(
    workspace: tuple[Settings, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, outer = workspace
    home = outer / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    script = RecordingScript(("write_file", {"path": "~/escaped.txt", "content": "pwned"}))

    await WorkerAgent(task=make_task(), client=FakeLLM(settings=settings, scripts=[script])).run()

    assert "Error:" in script.output_for("write_file")
    assert not (home / "escaped.txt").exists()
    assert files_under(outer) == {"ws/README.md"}


async def test_the_escape_refusal_reaches_the_model_so_it_can_retry(
    workspace: tuple[Settings, Path],
) -> None:
    """The worker keeps going after a refused write, so the tool call is recoverable."""
    settings, _ = workspace
    script = RecordingScript(
        ("write_file", {"path": "../escaped.txt", "content": "pwned"}),
        ("write_file", {"path": "y.txt", "content": "fine"}),
        summary="wrote y.txt",
    )

    result = await WorkerAgent(
        task=make_task(), client=FakeLLM(settings=settings, scripts=[script])
    ).run()

    assert "Error:" in script.outputs[0][1]
    assert result.success is True
    assert result.files_changed == ["y.txt"]
    assert (settings.workspace / "y.txt").read_text(encoding="utf-8") == "fine"


# --------------------------------------------------------------------------
# Idempotence / de-duplication
# --------------------------------------------------------------------------


async def test_rerunning_the_same_agent_does_not_double_append_files_changed(
    workspace: tuple[Settings, Path],
) -> None:
    settings, _ = workspace
    client = FakeLLM(
        settings=settings,
        scripts=[
            ToolScript(("write_file", {"path": "x.txt", "content": "one"}), summary="first"),
            ToolScript(("write_file", {"path": "x.txt", "content": "two"}), summary="second"),
        ],
    )
    agent = WorkerAgent(task=make_task(), client=client)

    first = await agent.run()
    second = await agent.run()

    assert first.files_changed == ["x.txt"]
    assert second.files_changed == ["x.txt"]


async def test_the_same_path_twice_in_one_run_is_deduplicated_and_last_write_wins(
    workspace: tuple[Settings, Path],
) -> None:
    settings, _ = workspace
    client = FakeLLM(
        settings=settings,
        scripts=[
            ToolScript(
                ("write_file", {"path": "x.txt", "content": "one"}),
                ("write_file", {"path": "x.txt", "content": "two"}),
                summary="rewrote x.txt",
            )
        ],
    )

    result = await WorkerAgent(task=make_task(), client=client).run()

    assert result.files_changed == ["x.txt"]
    assert (settings.workspace / "x.txt").read_text(encoding="utf-8") == "two"


# --------------------------------------------------------------------------
# edit_file preconditions
# --------------------------------------------------------------------------


async def test_edit_file_with_an_absent_old_string_is_a_model_visible_error(
    workspace: tuple[Settings, Path],
) -> None:
    settings, _ = workspace
    (settings.workspace / "x.txt").write_text("alpha\n", encoding="utf-8")
    script = RecordingScript(
        ("edit_file", {"path": "x.txt", "old_string": "missing", "new_string": "beta"})
    )

    result = await WorkerAgent(
        task=make_task(), client=FakeLLM(settings=settings, scripts=[script])
    ).run()

    message = script.output_for("edit_file")
    assert message.startswith("Error: ValueError:")
    assert "old_string was not found" in message
    assert (settings.workspace / "x.txt").read_text(encoding="utf-8") == "alpha\n"
    assert result.files_changed == []


async def test_an_ambiguous_old_string_without_replace_all_is_a_model_visible_error(
    workspace: tuple[Settings, Path],
) -> None:
    settings, _ = workspace
    (settings.workspace / "x.txt").write_text("dup\ndup\n", encoding="utf-8")
    script = RecordingScript(
        ("edit_file", {"path": "x.txt", "old_string": "dup", "new_string": "uniq"})
    )

    await WorkerAgent(task=make_task(), client=FakeLLM(settings=settings, scripts=[script])).run()

    message = script.output_for("edit_file")
    assert message.startswith("Error: ValueError:")
    assert "occurs 2 times" in message
    assert "replace_all" in message
    assert (settings.workspace / "x.txt").read_text(encoding="utf-8") == "dup\ndup\n"


async def test_replace_all_resolves_the_ambiguity(workspace: tuple[Settings, Path]) -> None:
    settings, _ = workspace
    (settings.workspace / "x.txt").write_text("dup\ndup\n", encoding="utf-8")
    script = RecordingScript(
        (
            "edit_file",
            {
                "path": "x.txt",
                "old_string": "dup",
                "new_string": "uniq",
                "replace_all": True,
            },
        )
    )

    result = await WorkerAgent(
        task=make_task(), client=FakeLLM(settings=settings, scripts=[script])
    ).run()

    assert "Replaced 2 occurrence(s)" in script.output_for("edit_file")
    assert (settings.workspace / "x.txt").read_text(encoding="utf-8") == "uniq\nuniq\n"
    assert result.files_changed == ["x.txt"]


# --------------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------------


async def test_an_unknown_tool_name_is_reported_rather_than_raised(
    workspace: tuple[Settings, Path],
) -> None:
    settings, _ = workspace
    script = RecordingScript(("rm_rf", {"path": "x.txt"}))

    await WorkerAgent(task=make_task(), client=FakeLLM(settings=settings, scripts=[script])).run()

    message = script.output_for("rm_rf")
    assert message.startswith("Error: unknown tool 'rm_rf'.")
    for name in ("read_file", "write_file", "edit_file", "list_files"):
        assert name in message


async def test_a_missing_required_argument_is_reported_rather_than_raised(
    workspace: tuple[Settings, Path],
) -> None:
    settings, _ = workspace
    script = RecordingScript(("write_file", {"content": "no path"}))

    await WorkerAgent(task=make_task(), client=FakeLLM(settings=settings, scripts=[script])).run()

    assert "'path' is required" in script.output_for("write_file")


async def test_the_worker_tool_schemas_name_four_workspace_relative_tools() -> None:
    names = [tool["name"] for tool in WORKER_TOOLS]

    assert names == ["read_file", "write_file", "edit_file", "list_files"]
    for tool in WORKER_TOOLS:
        assert tool["description"]
        assert tool["input_schema"]["type"] == "object"


async def test_read_file_and_list_files_go_through_the_workspace(
    workspace: tuple[Settings, Path],
) -> None:
    settings, _ = workspace
    (settings.workspace / "x.txt").write_text("alpha\n", encoding="utf-8")
    (settings.workspace / ".git").mkdir()
    (settings.workspace / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    script = RecordingScript(("list_files", {}), ("read_file", {"path": "x.txt"}))

    await WorkerAgent(task=make_task(), client=FakeLLM(settings=settings, scripts=[script])).run()

    listing = script.output_for("list_files")
    assert "README.md" in listing
    assert ".git" not in listing
    assert script.output_for("read_file") == "alpha\n"


async def test_list_files_outside_the_workspace_is_refused(
    workspace: tuple[Settings, Path],
) -> None:
    settings, _ = workspace
    script = RecordingScript(("list_files", {"directory": ".."}))

    await WorkerAgent(task=make_task(), client=FakeLLM(settings=settings, scripts=[script])).run()

    assert script.output_for("list_files").startswith("Error: ValueError:")
