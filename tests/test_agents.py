"""Contract tests for the three agents: planner, worker and tester.

Three properties are asserted here that the rest of the suite cannot reach:

* **Model output is untrusted.** A hostile plan payload is sanitised rather than trusted,
  a worker reply with no parseable JSON writes nothing, and a worker's file writes are
  confined to ``settings.workspace`` no matter what path the model asked for.
* **The runner's exit code is authoritative.** A model cannot turn a non-zero exit into a
  pass, and an empty ``failures`` list is back-filled from the runner output because the
  orchestrator rebuilds its focus from it.
* **``settings.workspace`` is the subprocess working directory.** The tester runs the suite
  of the workspace it was pointed at -- see :func:`test_workspace_is_the_runner_cwd`.

Every LLM interaction goes through the ``fake_llm`` fixture and every agent call is bounded
by :func:`bounded`, so no test here can reach the network or stall the suite.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

import pytest

from furrow.agents import planner as planner_module
from furrow.agents import tester as tester_module
from furrow.agents import worker as worker_module
from furrow.agents.planner import (
    MAX_LISTED_FILES,
    PLAN_PARSE_RETRIES,
    PREVIEW_CHARS,
    PlannerAgent,
)
from furrow.agents.prompts import PLANNER_PROMPT, WORKER_PROMPT
from furrow.agents.tester import (
    MAX_VERDICT_ATTEMPTS,
    RunnerOutcome,
    TesterAgent,
    runner_candidates,
)
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from tests.conftest import NESTED_RUN_GUARD

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

_T = TypeVar("_T")

#: Wall clock bound on every agent call made from this module.
AGENT_TIMEOUT = 5.0

#: Goal reused by the tester tests; short, and never a real repository path.
TESTER_GOAL = "make the widget tests pass"


async def bounded(awaitable: Awaitable[_T], timeout: float = AGENT_TIMEOUT) -> _T:
    """Await ``awaitable`` under a wall clock bound.

    An agent that looped forever or awaited a never-completing subprocess would otherwise
    hang the suite; here it fails its own test in ``timeout`` seconds instead.
    """
    return await asyncio.wait_for(awaitable, timeout)


def make_settings(workspace: Path, **overrides: Any) -> Settings:
    """Return settings pinned to ``workspace``, with no provider credentials."""
    return Settings(
        workspace=workspace,
        anthropic_api_key=None,
        openai_api_key=None,
        **overrides,
    )


def snapshot(root: Path) -> dict[str, Any]:
    """Return a byte-exact description of the tree under ``root``.

    Symlinks are recorded as links and never followed, so a workspace holding a link out of
    the tree can still be compared before and after without reading outside it.
    """
    entries: dict[str, Any] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        base = Path(dirpath)
        for name in [*dirnames, *filenames]:
            path = base / name
            key = str(path.relative_to(root))
            if path.is_symlink():
                entries[key] = ("symlink", os.readlink(path))
            elif path.is_dir():
                entries[key] = ("dir", None)
            else:
                entries[key] = ("file", path.read_bytes())
    return entries


def write_response(
    summary: str = "did the work",
    files: list[dict[str, Any]] | None = None,
) -> str:
    """Return a worker reply in the documented ``{"files": [...]}`` shape."""
    return json.dumps({"summary": summary, "files": files or []})


def pytest_workspace(tmp_path: Path) -> Path:
    """Create a workspace that :func:`runner_candidates` recognises as a pytest project."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    return workspace


def stub_runner(monkeypatch: pytest.MonkeyPatch, outcome: RunnerOutcome) -> list[Any]:
    """Replace the module level ``_spawn_runner`` seam and return its call log.

    This is the only supported way to exercise verdict handling: the real spawn would
    re-enter the runner for every case, which is neither deterministic nor fast.
    """
    seen: list[Any] = []

    async def _fake(argv: Any, workspace: Any) -> RunnerOutcome:
        seen.append((tuple(argv), Path(workspace)))
        return outcome

    monkeypatch.setattr(tester_module, "_spawn_runner", _fake)
    return seen


async def run_tester(
    fake_llm: Any,
    responses: list[str],
    workspace: Path,
    **overrides: Any,
) -> tuple[TestResult, Any]:
    """Run :class:`TesterAgent` against ``workspace`` and return ``(result, client)``."""
    client = fake_llm(responses, settings=make_settings(workspace, **overrides))
    result = await bounded(TesterAgent(client=client).run(TESTER_GOAL, []))
    return result, client


async def run_worker(
    fake_llm: Any,
    responses: list[str],
    workspace: Path,
    files: list[str] | None = None,
    **overrides: Any,
) -> tuple[str, Any]:
    """Run :class:`WorkerAgent` against ``workspace`` and return ``(summary, client)``."""
    client = fake_llm(responses, settings=make_settings(workspace, **overrides))
    task = TaskModel(id="1", description="add the widget module", files=files or [])
    summary = await bounded(WorkerAgent(task, client=client).run())
    return summary, client


# --------------------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------------------

#: A plan reply designed to break the orchestrator if it is trusted verbatim:
#: two non-object entries, a duplicate id, a blank id, a blank description, a task with no
#: id at all, and a dependency list carrying every kind of unusable edge.
HOSTILE_PLAN: dict[str, Any] = {
    "tasks": [
        "a bare string is not a task",
        7,
        {"id": "a", "description": "first task"},
        {"id": "a", "description": "duplicate id must lose"},
        {"id": "   ", "description": "blank id must be synthesised"},
        {"id": "orphan", "description": "   "},
        {"description": "no id at all"},
        {
            "id": "b",
            "description": "second task",
            "files": "single_scalar.py",
            "dependencies": ["a", "", "  ", "b", "ghost", "orphan", "a"],
        },
    ],
    "rationale": "hostile but recoverable",
}


async def test_plan_sanitises_hostile_payload(fake_llm: Any, tmp_path: Path) -> None:
    """Every unusable task and dependency edge is dropped, and ids are made unique."""
    client = fake_llm([json.dumps(HOSTILE_PLAN)], settings=make_settings(tmp_path))
    plan = await bounded(PlannerAgent(client=client).plan("ship the widget"))

    # Non-object entries and the blank-description entry are gone; ``orphan`` -- the id of a
    # dropped task -- is never a valid dependency target, so it is not an id either.
    assert [task.id for task in plan.tasks] == ["a", "5", "7", "b"]
    assert "orphan" not in {task.id for task in plan.tasks}
    # Duplicate ids resolve first-wins, so the losing description must be absent.
    assert [task.description for task in plan.tasks] == [
        "first task",
        "blank id must be synthesised",
        "no id at all",
        "second task",
    ]
    # Only the one legitimate edge survives: the blanks, the self-reference, the id unknown
    # to the plan, the id of a dropped task and the duplicate are all stripped.
    assert [task.dependencies for task in plan.tasks] == [[], [], [], ["a"]]
    assert plan.tasks[3].files == ["single_scalar.py"]
    assert plan.rationale == "hostile but recoverable"


@pytest.mark.parametrize(
    ("tasks", "expected_ids", "expected_dependencies", "expected_files"),
    [
        # A forward reference is legal: the edge is resolved once every id is known.
        (
            [
                {"id": "1", "description": "a", "dependencies": ["2"]},
                {"id": "2", "description": "b"},
            ],
            ["1", "2"],
            [["2"], []],
            [[], []],
        ),
        # A synthesised id that would collide with an explicit one gets a suffix.
        (
            [
                {"id": "2", "description": "explicit two"},
                {"description": "no id"},
            ],
            ["2", "2-2"],
            [[], []],
            [[], []],
        ),
        # A bare scalar dependency is coerced, a missing one is simply empty.
        (
            [
                {"id": "x", "description": "a", "dependencies": "x"},
                {"id": "y", "description": "b", "dependencies": None},
            ],
            ["x", "y"],
            [[], []],
            [[], []],
        ),
        # Blank file entries are dropped; a non-sequence ``files`` value is tolerated.
        (
            [
                {"id": "x", "description": "a", "files": ["  ", "keep.py", "", "also.py"]},
                {"id": "y", "description": "b", "files": 5},
            ],
            ["x", "y"],
            [[], []],
            [["keep.py", "also.py"], []],
        ),
        # Repeated identical edges collapse to one.
        (
            [{"id": "x", "description": "a", "dependencies": ["x", "x", "x"]}],
            ["x"],
            [[]],
            [[]],
        ),
    ],
    ids=[
        "forward-reference",
        "synthesised-id-collision",
        "scalar-dependencies",
        "files-coercion",
        "duplicate-edges",
    ],
)
async def test_plan_edge_cases(
    fake_llm: Any,
    tmp_path: Path,
    tasks: list[Any],
    expected_ids: list[str],
    expected_dependencies: list[list[str]],
    expected_files: list[list[str]],
) -> None:
    """Sanitisation keeps legal edges and synthesises unique ids without inventing order."""
    client = fake_llm([json.dumps({"tasks": tasks})], settings=make_settings(tmp_path))
    plan = await bounded(PlannerAgent(client=client).plan("ship the widget"))

    assert [task.id for task in plan.tasks] == expected_ids
    assert [task.dependencies for task in plan.tasks] == expected_dependencies
    assert [task.files for task in plan.tasks] == expected_files


@pytest.mark.parametrize(
    "response",
    ["I am sorry, I cannot help with that request.", "NOJSON " + "y" * 500],
    ids=["prose", "long-and-truncated"],
)
async def test_unparseable_plan_names_the_goal(
    fake_llm: Any, tmp_path: Path, response: str
) -> None:
    """An unusable reply raises a ValueError naming the goal and previewing the reply."""
    client = fake_llm([response, response], settings=make_settings(tmp_path))

    with pytest.raises(ValueError) as excinfo:
        await bounded(PlannerAgent(client=client).plan("ship the widget"))

    message = str(excinfo.value)
    assert "ship the widget" in message
    collapsed = " ".join(response.split())
    if len(collapsed) > PREVIEW_CHARS:
        # A long reply is truncated rather than pasted whole into the operator's log.
        assert collapsed[:PREVIEW_CHARS] in message
        assert collapsed not in message
        assert message.rstrip().endswith("...'")
    else:
        # A reply that fits the bound is quoted whole -- with no ellipsis to imply a cut.
        assert f"response preview: {collapsed!r}" in message
        assert not message.rstrip().endswith("...")


async def test_plan_recovers_from_one_unparseable_reply(fake_llm: Any, tmp_path: Path) -> None:
    """A retry is made after an unusable reply, and the retry budget is exactly bounded."""
    good = json.dumps({"tasks": [{"id": "1", "description": "do it"}], "rationale": "fine"})
    client = fake_llm(["not json at all", good], settings=make_settings(tmp_path))

    plan = await bounded(PlannerAgent(client=client).plan("ship the widget"))

    assert [task.id for task in plan.tasks] == ["1"]
    assert len(client.calls) == 1 + PLAN_PARSE_RETRIES
    retry_prompt = client.calls[-1]["prompt"]
    assert planner_module.JSON_ONLY_INSTRUCTION in retry_prompt
    assert "ship the widget" in retry_prompt


async def test_planner_uses_the_planner_model(fake_llm: Any, tmp_path: Path) -> None:
    """Model selection is explicit, not inherited from the client's generic default."""
    settings = make_settings(tmp_path, planner_model="planner-model-under-test")
    client = fake_llm([json.dumps({"tasks": [], "rationale": ""})], settings=settings)

    await bounded(PlannerAgent(client=client).plan("ship the widget"))

    assert [call["model"] for call in client.calls] == ["planner-model-under-test"]


async def test_planner_prompt_caps_listing_and_puts_the_goal_last(
    fake_llm: Any, tmp_path: Path
) -> None:
    """The workspace listing is bounded, truncated with a note, and precedes the goal."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    total = MAX_LISTED_FILES + 25
    hidden = 25
    for index in range(total):
        (workspace / f"file_{index:04d}.txt").write_text("x")

    goal = "ship the widget"
    client = fake_llm([json.dumps({"tasks": [], "rationale": ""})], settings=make_settings(workspace))
    await bounded(PlannerAgent(client=client).plan(goal))

    prompt = client.prompts[0]
    # The listing is two lines: a header naming the workspace and the count, then the joined
    # paths on the line below it. Anything else would silently parse an empty file list.
    lines = prompt.splitlines()
    position = next(i for i, line in enumerate(lines) if line.startswith("Workspace files"))
    header, names = lines[position], lines[position + 1]
    listed = names.split(", ")
    assert len(listed) == MAX_LISTED_FILES
    assert listed[0] == "file_0000.txt"
    assert listed[-1] == f"file_{MAX_LISTED_FILES - 1:04d}.txt"
    assert f"({total} total, {hidden} more not shown)" in header

    # The goal goes last, where attention is strongest, and the rules are carried along.
    trimmed = prompt.rstrip()
    assert trimmed.endswith(f"Goal: {goal}")
    assert trimmed.rindex(goal) > trimmed.rindex(listed[-1])
    assert PLANNER_PROMPT in prompt
    assert planner_module.DEPENDENCY_RULES in prompt
    assert "dependencies" in prompt
    assert "concurrently" in prompt.lower()


async def test_planner_prompt_reports_an_empty_workspace(fake_llm: Any, tmp_path: Path) -> None:
    """A workspace with no files still reaches the prompt, as an explicit statement."""
    workspace = tmp_path / "empty"
    workspace.mkdir()
    client = fake_llm([json.dumps({"tasks": [], "rationale": ""})], settings=make_settings(workspace))

    await bounded(PlannerAgent(client=client).plan("ship the widget"))

    assert "contains no files" in client.prompts[0]
    assert client.prompts[0].rstrip().endswith("Goal: ship the widget")


# --------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------


async def test_worker_writes_declared_files(fake_llm: Any, tmp_path: Path) -> None:
    """A reply in the documented shape lands on disk and is named in the summary."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    response = write_response(
        summary="added the widget module and its readme",
        files=[
            {"path": "pkg/mod.py", "content": "VALUE = 1\n"},
            {"path": "README.md", "content": "# widget\n"},
        ],
    )

    summary, _ = await run_worker(fake_llm, [response], workspace, files=["pkg/mod.py"])

    assert (workspace / "pkg" / "mod.py").read_text() == "VALUE = 1\n"
    assert (workspace / "README.md").read_text() == "# widget\n"
    assert "wrote 2 of 2 requested file(s)" in summary
    assert "pkg/mod.py" in summary
    assert "README.md" in summary
    assert "added the widget module and its readme" in summary


@pytest.mark.parametrize(
    "response",
    ["", "I have completed the task as requested.", "```\nnot json at all\n```", "[1,2,3]"],
    ids=["empty", "prose", "fenced-non-json", "json-array"],
)
async def test_worker_writes_nothing_without_parseable_json(
    fake_llm: Any, tmp_path: Path, response: str
) -> None:
    """A reply that describes no file changes leaves the workspace byte-identical."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "keep.txt").write_text("keep")
    before = snapshot(workspace)

    summary, client = await run_worker(fake_llm, [response], workspace, files=["keep.txt"])

    assert snapshot(workspace) == before == {"keep.txt": ("file", b"keep")}
    assert "no file changes" in summary
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "path",
    ["../escape.py", "a/b/../../../escape.py", "/etc/passwd", "~/x", "C:\\x", "..\\win_escape.py"],
    ids=["parent", "nested-parent", "absolute", "home", "drive", "windows-parent"],
)
async def test_worker_rejects_paths_outside_the_workspace(
    fake_llm: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """No reply may write outside ``settings.workspace``, however the path is spelled."""
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "keep.txt").write_text("keep")

    summary, _ = await run_worker(fake_llm, [write_response(files=[{"path": path, "content": "pwned"}])], workspace)

    assert "wrote 0 of 1 requested file(s)" in summary
    assert "refused 1 unsafe path(s)" in summary
    assert snapshot(workspace) == {"keep.txt": ("file", b"keep")}
    # Nothing landed in the parent of the workspace either.
    assert set(os.listdir(tmp_path)) == {"ws"}


async def test_worker_rejects_a_symlink_pointing_out_of_the_workspace(
    fake_llm: Any, tmp_path: Path
) -> None:
    """A symlink inside the workspace must not become a tunnel out of it."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)

    summary, _ = await run_worker(
        fake_llm, [write_response(files=[{"path": "link/pwned.py", "content": "pwned"}])], workspace
    )

    assert "refused 1 unsafe path(s)" in summary
    assert "path escapes the workspace" in summary
    assert sorted(os.listdir(outside)) == ["secret.txt"]
    assert snapshot(workspace) == {"link": ("symlink", str(outside))}


@pytest.mark.parametrize(
    ("path", "expected_children"),
    [
        ("mod.py", ["mod.py"]),
        ("pkg/sub/deep.py", ["pkg"]),
        (".hidden", [".hidden"]),
        ("name with spaces.txt", ["name with spaces.txt"]),
        # A backslash is an ordinary character in a POSIX filename, so over-restricting it
        # would reject a legitimate file. Asserted explicitly, not just accepted.
        ("weird\\name.txt", ["weird\\name.txt"]),
    ],
    ids=["plain", "nested", "dotfile", "spaces", "backslash-in-name"],
)
async def test_worker_accepts_legitimate_paths(
    fake_llm: Any, tmp_path: Path, path: str, expected_children: list[str]
) -> None:
    """Ordinary in-workspace paths, including a real backslash in a file name, are written."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    summary, _ = await run_worker(fake_llm, [write_response(files=[{"path": path, "content": "ok\n"}])], workspace)

    assert "wrote 1 of 1 requested file(s)" in summary
    assert "refused" not in summary
    assert (workspace / path).read_text() == "ok\n"
    assert sorted(os.listdir(workspace)) == expected_children


async def test_worker_writes_are_confined_to_settings_workspace(
    fake_llm: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legal and illegal paths in one reply: the legal one lands, the escape is refused."""
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "ws"
    response = write_response(
        files=[
            {"path": "pkg/ok.py", "content": "ok\n"},
            {"path": "../outside.py", "content": "pwned\n"},
        ],
    )

    summary, _ = await run_worker(fake_llm, [response], workspace)

    assert "wrote 1 of 2 requested file(s)" in summary
    assert "refused 1 unsafe path(s)" in summary
    assert (workspace / "pkg" / "ok.py").read_text() == "ok\n"
    assert set(os.listdir(tmp_path)) == {"ws"}


async def test_worker_prompt_requests_the_json_shape_and_uses_the_worker_model(
    fake_llm: Any, tmp_path: Path
) -> None:
    """The reply contract is stated last and the worker model is selected explicitly."""
    settings = make_settings(tmp_path, worker_model="worker-model-under-test")
    client = fake_llm([write_response()], settings=settings)

    await bounded(
        WorkerAgent(
            TaskModel(id="1", description="add the widget module", files=["pkg/mod.py"]),
            client=client,
        ).run()
    )

    prompt = client.prompts[0]
    assert prompt.rstrip().endswith(worker_module.FILE_CHANGE_SHAPE.rstrip())
    assert '"files"' in prompt
    assert '"path"' in prompt
    assert '"content"' in prompt
    assert WORKER_PROMPT in prompt
    assert "add the widget module" in prompt
    assert "pkg/mod.py" in prompt
    assert [call["model"] for call in client.calls] == ["worker-model-under-test"]


# --------------------------------------------------------------------------------------
# tester
# --------------------------------------------------------------------------------------


async def test_workspace_is_the_runner_cwd(
    fake_llm: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The suite that runs is the one in ``settings.workspace``, not the harness's own.

    The workspace holds a pytest marker and a single deliberately failing test whose
    assertion message carries :data:`MARKER`. The model claims the run passed, so the only
    way the verdict can come out failed -- and the only way the marker can appear at all --
    is if the subprocess really ran inside the workspace.
    """
    marker = "FURROW_WORKSPACE_CWD_MARKER"
    # If the runner inherited this process's working directory it would collect this very
    # suite, so the guard makes that recursion fail loudly instead of hanging.
    monkeypatch.setenv(NESTED_RUN_GUARD, "workspace-cwd-test")
    # The autouse fixture clamps this to 2s so a provoked timeout cannot stall the suite;
    # this test needs the real thing to *succeed*, and it takes well under a second.
    monkeypatch.setattr(tester_module, "TEST_RUN_TIMEOUT_SECONDS", 30.0)

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (workspace / "test_widget_sample.py").write_text(
        "def test_widget_count():\n"
        f"    assert False, {marker!r}\n"
    )

    result, _ = await run_tester(
        fake_llm,
        ['{"passed": true, "summary": "all green", "failures": []}'],
        workspace,
    )
    await asyncio.sleep(0)

    report = "\n".join([result.summary, *result.failures])
    assert result.passed is False
    assert marker in report
    assert "test_widget_sample" in report
    # Nothing from this repository's own suite may appear in the report.
    assert "test_plan_parse" not in report
    assert "test_core" not in report


@pytest.mark.parametrize(
    "claim",
    [
        '{"passed": true, "summary": "looks green", "failures": []}',
        '{"passed": "passed", "summary": "looks green", "failures": []}',
        '{"passed": 1, "summary": "looks green", "failures": []}',
        "Everything passed, nothing to report.",
    ],
    ids=["true", "string-truthy", "numeric", "no-json"],
)
async def test_nonzero_exit_cannot_be_overridden_by_the_model(
    fake_llm: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, claim: str
) -> None:
    """The runner's exit code wins over any claim of success."""
    workspace = pytest_workspace(tmp_path)
    outcome = RunnerOutcome(argv=("pytest", "-q"), returncode=1, output="1 failed in 0.01s")
    stub_runner(monkeypatch, outcome)

    result, _ = await run_tester(fake_llm, [claim], workspace)

    assert result.passed is False


@pytest.mark.parametrize(
    ("runner_output", "expected_fragment"),
    [
        ("E   AssertionError: the widget count is wrong\n1 failed in 0.01s", "widget count"),
        ("not ok 1 - widget should render\n", "widget should render"),
        # ``go test`` marks a failing case with a bare ``--- FAIL:`` header, which carries no
        # marker worth mining; the ``panic:`` line underneath it is what names the failure.
        (
            (
                "--- FAIL: TestWidgetCount (0.00s)\n"
                "panic: runtime error: index out of range [3] with length 3 [recovered]\n"
                "\tpanic: runtime error: index out of range [3] with length 3\n"
                "FAIL\texample.com/widget\t0.004s\n"
            ),
            "index out of range",
        ),
    ],
    ids=["pytest", "node", "go"],
)
async def test_empty_failures_are_backfilled_from_runner_output(
    fake_llm: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner_output: str,
    expected_fragment: str,
) -> None:
    """An empty ``failures`` list is replaced with mined output.

    The orchestrator rebuilds its next focus from ``failures``, so an empty list would leave
    it with nothing to act on even though the suite failed.
    """
    workspace = pytest_workspace(tmp_path)
    outcome = RunnerOutcome(argv=("pytest", "-q"), returncode=1, output=runner_output)
    stub_runner(monkeypatch, outcome)

    result, _ = await run_tester(
        fake_llm, ['{"passed": true, "summary": "all green", "failures": []}'], workspace
    )

    assert result.passed is False
    assert result.failures
    assert any(expected_fragment in failure for failure in result.failures)
    assert result.summary


@pytest.mark.parametrize(
    "verdict",
    ["[1,2,3]", "the suite is green", "3", "null"],
    ids=["json-array", "prose", "json-scalar", "json-null"],
)
async def test_non_object_verdict_never_raises(
    fake_llm: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str
) -> None:
    """A reply that is not a JSON object is retried, then reported -- never propagated."""
    workspace = pytest_workspace(tmp_path)
    outcome = RunnerOutcome(argv=("pytest", "-q"), returncode=1, output="boom")
    stub_runner(monkeypatch, outcome)

    result, client = await run_tester(fake_llm, [verdict], workspace)

    assert isinstance(result, TestResult)
    assert result.passed is False
    assert len(client.calls) == MAX_VERDICT_ATTEMPTS


@pytest.mark.parametrize(
    "verdict",
    [
        '```json\n{"passed": true, "summary": "green", "failures": []}\n```',
        'Here is the verdict:\n{"passed": true, "summary": "green", "failures": []}\nHope it helps.',
        '{"summary": "model says the run failed", "passed": false, "failures": ["only failure"]}',
    ],
    ids=["fenced", "prose-wrapped", "model-failure-believed"],
)
async def test_wrapped_verdicts_are_parsed(
    fake_llm: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str
) -> None:
    """Fenced or prose-wrapped verdicts parse; a reported failure on a green run is believed."""
    workspace = pytest_workspace(tmp_path)
    outcome = RunnerOutcome(argv=("pytest", "-q"), returncode=0, output="3 passed in 0.05s")
    stub_runner(monkeypatch, outcome)

    result, client = await run_tester(fake_llm, [verdict], workspace)

    assert len(client.calls) == 1
    assert result.passed is (verdict.count('"passed": true') == 1)
    assert result.summary
    if result.passed is False:
        assert result.failures == ["only failure"]


async def test_missing_workspace_is_reported_without_any_llm_call(
    fake_llm: Any, tmp_path: Path
) -> None:
    """A workspace that does not exist fails the result rather than raising."""
    result, client = await run_tester(fake_llm, ["unused"], tmp_path / "does-not-exist")

    assert result.passed is False
    assert "does not exist" in result.summary
    assert result.failures
    assert client.calls == []


async def test_unrecognised_workspace_makes_no_llm_call(
    fake_llm: Any, tmp_path: Path
) -> None:
    """No recognised runner means nothing is run and the model is never consulted."""
    workspace = tmp_path / "notes"
    workspace.mkdir()
    (workspace / "README.md").write_text("no runner here")

    result, client = await run_tester(fake_llm, ["unused"], workspace)

    assert result.passed is False
    assert "No recognised test runner" in result.summary
    assert result.failures
    assert client.calls == []
    assert runner_candidates(workspace) == []


@pytest.mark.parametrize(
    ("marker", "extra", "expected_head"),
    [
        ("pyproject.toml", None, (sys.executable, "-m", "pytest", "-q")),
        ("pytest.ini", None, (sys.executable, "-m", "pytest", "-q")),
        ("setup.cfg", None, (sys.executable, "-m", "pytest", "-q")),
        ("package.json", "package-lock.json", ("npm",)),
        ("package.json", "yarn.lock", ("yarn",)),
        ("package.json", "pnpm-lock.yaml", ("pnpm",)),
        ("package.json", None, ("npm",)),
        ("Cargo.toml", None, ("cargo",)),
        ("go.mod", None, ("go",)),
        ("notes.txt", None, ()),
    ],
    ids=[
        "pyproject",
        "pytest-ini",
        "setup-cfg",
        "npm",
        "yarn",
        "pnpm",
        "npm-default",
        "cargo",
        "go",
        "unrecognised",
    ],
)
async def test_runner_is_chosen_from_marker_files_before_anything_runs(
    fake_llm: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
    extra: str | None,
    expected_head: tuple[str, ...],
) -> None:
    """The runner comes from the project's own markers; an unknown project runs nothing."""
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / marker).write_text("[tool.pytest.ini_options]\n")
    if extra is not None:
        (workspace / extra).write_text("{}\n")

    calls: list[Any] = []

    async def _should_not_run(argv: Any, workspace_arg: Any) -> RunnerOutcome:
        calls.append((tuple(argv), Path(workspace_arg)))
        return RunnerOutcome(argv=tuple(argv), returncode=0, output="1 passed")

    monkeypatch.setattr(tester_module, "_spawn_runner", _should_not_run)
    client = fake_llm(['{"passed": true, "summary": "green", "failures": []}'], settings=make_settings(workspace))

    await bounded(TesterAgent(client=client).run(TESTER_GOAL, []))

    candidates = runner_candidates(workspace)
    if expected_head:
        assert candidates, f"no runner candidates for marker {marker!r}"
        assert candidates[0][: len(expected_head)] == expected_head
        # The workspace is handed to the spawn seam as the working directory.
        assert [call[1] for call in calls] == [workspace.resolve()] * len(calls)
    else:
        assert candidates == []
        assert calls == []
        assert client.calls == []