"""Tests for the three agents: :class:`PlannerAgent`, :class:`TesterAgent` and
:class:`WorkerAgent`.

Three failure modes are pinned down here, because each one silently corrupted a
cycle rather than crashing it:

* the planner could not parse a fenced-JSON reply behind a prose preamble, which
  killed the cycle outright;
* the tester could report a *successful* test run when no runner was ever
  executed, letting the loop declare a goal verified that was never tested;
* the worker writes model-authored paths, so path resolution is a security
  boundary: an LLM must not be able to write outside the workspace.

Everything runs offline against :class:`tests.conftest.FakeLLM`. The tester is
always given an explicit ``test_command`` and ``LINT_COMMANDS`` is always patched
out, so no test can shell out to the repository's real ``pytest``/``ruff``/
``mypy``.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

import furrow.agents.tester as tester_module
from furrow.agents.planner import DEFAULT_RATIONALE, HARD_MAX_TASKS, PlannerAgent
from furrow.agents.tester import MAX_OUTPUT_CHARS, TesterAgent, _truncate
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from tests.conftest import FakeLLM, plan_reply, worker_reply

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from furrow.config import Settings

#: Written by the in-workspace worker test and compared byte for byte.
EXACT_CONTENT = "line one\nline two\n\tindented — unicode ✓\n"

#: Just under the tester's output cap, so the emitted output is definitely larger.
NOISE_CHARS = 9000


# --------------------------------------------------------------------- helpers


def _task(task_id: str = "1", description: str = "do the thing", **kwargs: object) -> TaskModel:
    """A minimal :class:`TaskModel`; only the id and description are ever needed."""
    return TaskModel(id=task_id, description=description, **kwargs)  # type: ignore[arg-type]


def _planner(reply: str, settings: Settings) -> tuple[PlannerAgent, FakeLLM]:
    """A planner whose only model is ``settings.planner_model``, answering ``reply``."""
    fake = FakeLLM({settings.planner_model: reply}, settings=settings, workspace=settings.workspace)
    return PlannerAgent(client=fake, settings=settings), fake


def _tester(
    reply: str,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TesterAgent, FakeLLM]:
    """A tester whose only model is ``settings.tester_model``, answering ``reply``.

    ``LINT_COMMANDS`` is emptied unconditionally: a green test command would
    otherwise be joined by a real ``ruff check .`` and ``mypy .`` over the whole
    repository, which is slow, non-deterministic and unrelated to these tests.
    """
    monkeypatch.setattr(tester_module, "LINT_COMMANDS", ())
    fake = FakeLLM({settings.tester_model: reply}, settings=settings, workspace=settings.workspace)
    return TesterAgent(client=fake, settings=settings), fake


def _worker(reply: str, settings: Settings) -> tuple[WorkerAgent, FakeLLM]:
    """A worker for a single task, with ``reply`` as its canned model answer."""
    fake = FakeLLM({settings.worker_model: reply}, settings=settings, workspace=settings.workspace)
    return WorkerAgent(task=_task(), client=fake, settings=settings), fake


def _py(code: str) -> list[str]:
    """A test command running ``code`` with the *current* interpreter."""
    return [sys.executable, "-c", code]


# --------------------------------------------------------------------- planner


async def test_plan_parses_fenced_json_behind_a_prose_preamble(make_settings: Callable[..., Settings]) -> None:
    """Regression: a ```json fence plus preamble used to raise ``ValueError``.

    ``extract_json`` is the fix, and models really do wrap JSON that way, so the
    planner must strip the fence instead of aborting the whole cycle.
    """
    settings = make_settings()
    reply = plan_reply({"id": "1", "description": "add the endpoint"})
    assert "```json" in reply, "the fixture must exercise the fence-stripping path"
    assert reply.lstrip().startswith("Here is"), "the fixture must exercise the preamble path"

    agent, fake = _planner(reply, settings)
    plan = await agent.plan("add an endpoint")

    assert [task.id for task in plan.tasks] == ["1"]
    assert plan.tasks[0].description == "add the endpoint"
    assert fake.calls[0][0] == settings.planner_model


async def test_plan_makes_duplicate_task_ids_unique(make_settings: Callable[..., Settings]) -> None:
    """Regression: duplicate ids made the dependency graph ambiguous.

    ``TaskModel.id`` is what the orchestrator keys on, so a repeated id would
    collapse two distinct tasks into one. The second occurrence gets ``-2``.
    """
    settings = make_settings()
    agent, _ = _planner(
        plan_reply(
            {"id": "a", "description": "first"},
            {"id": "a", "description": "second"},
            {"id": "a", "description": "third"},
        ),
        settings,
    )

    plan = await agent.plan("goal")

    assert [task.id for task in plan.tasks] == ["a", "a-2", "a-3"]
    assert [task.description for task in plan.tasks] == ["first", "second", "third"]


async def test_plan_strips_a_task_dependency_on_itself(make_settings: Callable[..., Settings]) -> None:
    """Regression: a self-dependency is an unschedulable cycle.

    Such a task can never run, so the dependency is removed rather than left to
    block the task forever. Task ``c`` is a real sibling task, so its dependency
    edge genuinely survives and the test isolates the self-reference rule alone
    (a dependency on an *absent* id is dropped for a different reason -- see
    :func:`test_plan_drops_dependencies_on_absent_tasks`).
    """
    settings = make_settings()
    agent, _ = _planner(
        plan_reply(
            {"id": "b", "description": "self referential", "dependencies": ["b", "c"]},
            {"id": "c", "description": "producer"},
        ),
        settings,
    )

    plan = await agent.plan("goal")

    by_id = {task.id: task for task in plan.tasks}
    assert by_id["b"].dependencies == ["c"]


async def test_plan_drops_dependencies_on_absent_tasks(make_settings: Callable[..., Settings]) -> None:
    """A dependency on an id the plan does not contain is discarded.

    ``TaskModel.dependencies`` is resolved by id lookup at run time, so a
    dangling reference would wait forever for a producer that does not exist.
    """
    settings = make_settings()
    agent, _ = _planner(
        plan_reply(
            {"id": "c", "description": "consumer", "dependencies": ["ghost", "b"]},
            {"id": "b", "description": "producer"},
        ),
        settings,
    )

    plan = await agent.plan("goal")

    by_id = {task.id: task for task in plan.tasks}
    assert by_id["c"].dependencies == ["b"]
    assert "ghost" not in by_id["c"].dependencies


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"tasks": [{"id": "1", "description": "d"}]},
            id="missing",
        ),
        pytest.param(
            {"tasks": [{"id": "1", "description": "d"}], "rationale": "   "},
            id="blank",
        ),
    ],
)
async def test_plan_falls_back_to_a_default_rationale(
    make_settings: Callable[..., Settings],
    payload: dict[str, object],
) -> None:
    """``Plan.rationale`` is a required pydantic field.

    A reply without one must not blow up the cycle on a validation error, so the
    planner substitutes a documented default.
    """
    settings = make_settings()
    agent, _ = _planner(json.dumps(payload), settings)

    plan = await agent.plan("goal")

    assert plan.rationale == DEFAULT_RATIONALE
    assert plan.rationale.strip()


async def test_plan_clamps_the_task_count_to_max_parallel_tasks(
    make_settings: Callable[..., Settings],
) -> None:
    """A plan longer than the configured fan-out is truncated.

    The workers run concurrently, so the clamp is what keeps one hallucinated
    forty-task plan from fanning out forty subprocesses and LLM calls at once.
    """
    settings = make_settings(max_parallel_tasks=2)
    agent, _ = _planner(
        plan_reply(*({"id": str(index), "description": f"step {index}"} for index in range(9))),
        settings,
    )

    plan = await agent.plan("goal")

    assert len(plan.tasks) == 2
    assert [task.id for task in plan.tasks] == ["0", "1"]


async def test_plan_clamps_the_task_count_to_the_hard_ceiling(
    make_settings: Callable[..., Settings],
) -> None:
    """A generous ``max_parallel_tasks`` cannot lift the hard ceiling.

    ``HARD_MAX_TASKS`` is the agent's own backstop, independent of settings, so a
    misconfigured value cannot re-open the fan-out hole.
    """
    settings = make_settings(max_parallel_tasks=99)
    agent, _ = _planner(
        plan_reply(*({"id": str(index), "description": f"step {index}"} for index in range(9))),
        settings,
    )

    plan = await agent.plan("goal")

    assert settings.max_parallel_tasks > HARD_MAX_TASKS
    assert len(plan.tasks) == HARD_MAX_TASKS


# ---------------------------------------------------------------------- tester


async def test_a_test_command_that_cannot_run_is_never_a_pass(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: reported success when no runner was found.

    The tester used to return the string ``"No test runner found."`` and let the
    model read a verdict out of it, so a project with no runnable suite looked
    green and the loop declared victory on an unverified goal. Nothing ran here,
    so the result must be an explicit failure, and the model must not be asked to
    opine at all (zero LLM calls proves the verdict was not delegated).
    """
    settings = make_settings()
    agent, fake = _tester("{}", settings, monkeypatch)

    result = await agent.run(
        "goal",
        [_task()],
        test_command=["definitely-not-a-runner-xyz"],
    )

    assert result.passed is False
    assert "UNVERIFIED" in result.summary.upper()
    assert "definitely-not-a-runner-xyz" in result.summary
    assert result.failures, "an unverified cycle must carry a failure for the orchestrator"
    assert "unverified" in result.failures[0].lower()
    assert fake.calls == [], "no run means there is nothing for a model to summarise"


async def test_runner_discovery_finding_nothing_is_unverified(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same false-pass regression via the default runner list.

    The explicit-command case above is not the realistic one: in a repo with no
    pytest/npm/cargo installed, ``TEST_COMMANDS`` is consulted and every
    candidate is missing. That must still fail, not fall through to a pass.
    """
    settings = make_settings()
    monkeypatch.setattr(
        tester_module,
        "TEST_COMMANDS",
        (("no-such-runner-abc",), ("no-such-runner-def",)),
    )
    agent, fake = _tester("{}", settings, monkeypatch)

    result = await agent.run("goal", [_task()])

    assert result.passed is False
    assert "UNVERIFIED" in result.summary.upper()
    assert "no-such-runner-abc" in result.summary
    assert result.failures
    assert fake.calls == []


async def test_a_nonzero_exit_beats_a_model_claiming_success(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the exit code is authoritative, the model only summarises.

    A model that reports ``"passed": true`` for a suite that exited 1 would
    otherwise flip a red run green, which is the whole loop's safety net.
    """
    settings = make_settings()
    agent, _ = _tester(
        json.dumps({"passed": True, "summary": "everything looks good", "failures": []}),
        settings,
        monkeypatch,
    )

    result = await agent.run(
        "goal",
        [_task()],
        test_command=_py("import sys; sys.stderr.write('boom: 1 failed\\n'); sys.exit(1)"),
    )

    assert result.passed is False
    assert "everything looks good" in result.summary, "the model's prose is still reported"
    assert result.failures, "a failed run must list its failures"
    assert "exit 1" in result.failures[0]
    assert "boom: 1 failed" in result.failures[0]


async def test_a_zero_exit_beats_a_model_claiming_failure(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verdict is symmetric: a green run stays green.

    The model may add prose, but it must not be able to invent failures either,
    nor inject its own failure entries into the list the orchestrator consumes.
    """
    settings = make_settings()
    agent, _ = _tester(
        json.dumps({"passed": False, "summary": "I think something broke", "failures": ["invented"]}),
        settings,
        monkeypatch,
    )

    result = await agent.run(
        "goal",
        [_task()],
        test_command=_py("print('1 passed')"),
    )

    assert result.passed is True
    assert "I think something broke" in result.summary
    assert result.failures == []


def test_long_output_is_truncated_from_the_head() -> None:
    """The tail is kept because that is where the assertion failure lives.

    A suite that fails on its last test would otherwise be truncated into a pass
    by keeping the head of its output.
    """
    head, tail = "x" * NOISE_CHARS, "TAIL_MARKER_END"

    text, truncated = _truncate(head + tail)

    assert truncated is True
    assert text.endswith(tail)
    assert "earlier characters truncated" in text
    assert len(text) <= MAX_OUTPUT_CHARS + 100, "the cap must still bound the prompt"

    # Output already inside the cap is passed through untouched and unflagged.
    short = "1 passed"
    assert _truncate(short) == (short, False)


async def test_the_truncated_tail_reaches_the_tester_prompt(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: the model summarising a truncated run still sees the failure.

    The head/tail markers are assembled at runtime on purpose -- the command line
    is echoed into the prompt as ``Commands run: ...``, so literal markers in the
    source would satisfy the assertions no matter what was truncated.
    """
    settings = make_settings()
    agent, fake = _tester(json.dumps({"passed": True, "summary": "green"}), settings, monkeypatch)
    code = (
        "import sys\n"
        "sys.stdout.write('HEAD_' + 'MARKER_START')\n"
        f"sys.stdout.write('x' * {NOISE_CHARS})\n"
        "sys.stdout.write('TAIL_' + 'MARKER_END\\n')\n"
    )

    result = await agent.run("goal", [_task()], test_command=_py(code))
    prompt = fake.calls[0][1]

    assert result.passed is True
    assert "TAIL_MARKER_END" in prompt, "the failure end of the output must survive"
    assert "HEAD_MARKER_START" not in prompt, "the head is what truncation is allowed to drop"
    assert "truncated" in prompt


# ---------------------------------------------------------------------- worker


async def test_worker_writes_an_in_workspace_edit_byte_for_byte(
    make_settings: Callable[..., Settings],
) -> None:
    """The worker used to describe its change in prose and write nothing at all.

    A task is only done if the file changed, so a normal relative path must be
    written exactly as the model returned it -- no trailing-newline fixups, no
    reformatting, no encoding surprises.
    """
    settings = make_settings()
    agent, fake = _worker(
        worker_reply(edits=[{"path": "pkg/mod.py", "content": EXACT_CONTENT}]),
        settings,
    )
    expected = (settings.workspace / "pkg" / "mod.py").resolve()

    result = await agent.run()

    assert expected.is_file()
    assert expected.read_bytes() == EXACT_CONTENT.encode()
    assert fake.writes == [(expected, EXACT_CONTENT)], "recording and disk must agree"
    assert "wrote 1 file(s)" in result
    assert "pkg/mod.py" in result


async def test_worker_rejects_an_edit_that_traverses_out_of_the_workspace(
    tmp_path: Path,
    make_settings: Callable[..., Settings],
) -> None:
    """Security boundary: ``..`` must not escape the workspace.

    The workspace is the sandbox; a model that could write ``../escape.txt``
    would be editing the parent directory the orchestrator never granted it.
    """
    settings = make_settings(workspace=tmp_path)
    escape_target = tmp_path.parent / "escape.txt"
    agent, fake = _worker(
        worker_reply(edits=[{"path": "../escape.txt", "content": "owned"}]),
        settings,
    )

    with pytest.raises(RuntimeError, match="no usable edits"):
        await agent.run()

    assert not escape_target.exists(), "a traversal write escaped the workspace"
    assert fake.writes == []


async def test_worker_rejects_an_absolute_path_outside_the_workspace(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    make_settings: Callable[..., Settings],
) -> None:
    """Security boundary: an absolute path outside the workspace is rejected.

    The target is a *sibling* temp directory, never a real system path: this
    suite runs as root, so a regression here would have to be a genuine escape to
    be noticed -- and it must not be able to clobber the box while proving it.
    """
    outside = tmp_path_factory.mktemp("outside_workspace")
    escape_target = outside / "escape_absolute.txt"
    settings = make_settings(workspace=tmp_path)
    agent, fake = _worker(
        worker_reply(edits=[{"path": str(escape_target), "content": "owned"}]),
        settings,
    )

    with pytest.raises(RuntimeError, match="no usable edits"):
        await agent.run()

    assert not escape_target.exists(), "an absolute write escaped the workspace"
    assert fake.writes == []


async def test_worker_rejects_an_escape_through_a_symlinked_directory(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    make_settings: Callable[..., Settings],
) -> None:
    """Security boundary: a symlink inside the workspace cannot be used to leave it.

    Every path component is resolved before the containment check, so a symlinked
    directory pointing out of the workspace is rejected even though the requested
    path is textually inside it.
    """
    outside = tmp_path_factory.mktemp("symlink_target")
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    escape_target = outside / "escape_symlink.txt"

    settings = make_settings(workspace=tmp_path)
    agent, fake = _worker(
        worker_reply(edits=[{"path": "link/escape_symlink.txt", "content": "owned"}]),
        settings,
    )

    with pytest.raises(RuntimeError, match="no usable edits"):
        await agent.run()

    assert not escape_target.exists(), "a symlinked write escaped the workspace"
    assert fake.writes == []


def test_resolve_workspace_path_rejects_system_paths(
    tmp_path: Path,
    make_settings: Callable[..., Settings],
) -> None:
    """Pure path resolution, no writes: even a real system file is refused.

    This is the non-destructive view of the same boundary as the tests above, so
    ``/etc/passwd`` is only ever resolved and rejected, never written.
    """
    settings = make_settings(workspace=tmp_path)
    agent, _ = _worker(worker_reply(), settings)

    assert agent.resolve_workspace_path("/etc/passwd") is None
    assert agent.resolve_workspace_path("../../elsewhere.txt") is None
    assert agent.resolve_workspace_path("  ") is None
    # ...while a legitimate path inside the workspace still resolves.
    assert agent.resolve_workspace_path("pkg/mod.py") == (tmp_path / "pkg" / "mod.py").resolve()


def test_parse_response_degrades_gracefully_on_prose(
    make_settings: Callable[..., Settings],
) -> None:
    """Layer 1 of 2: the parser never raises, it reports.

    A model that answers with plain English must not crash the worker; the reply
    is turned into a summary with no edits and an issue, so the caller decides
    what to do.
    """
    settings = make_settings()
    agent, _ = _worker("I could not do the task, sorry.", settings)

    parsed = agent.parse_response("I could not do the task, sorry.")

    assert isinstance(parsed, dict)
    assert parsed["edits"] == []
    assert parsed["issues"], "an unparsable reply must be flagged"
    assert "unparsable" in parsed["issues"][0]


async def test_run_raises_when_the_worker_produced_no_usable_edits(
    tmp_path: Path,
    make_settings: Callable[..., Settings],
) -> None:
    """Layer 2 of 2: ``run()`` deliberately raises on the degenerate case.

    Silently returning "did nothing" let a cycle advance as if work had landed.
    The orchestrator runs the workers in
    ``asyncio.gather(..., return_exceptions=True)``, so this ``RuntimeError`` is
    exactly what turns into ``task.status = "failed"``.
    """
    settings = make_settings(workspace=tmp_path)
    agent, fake = _worker("I could not do the task, sorry.", settings)

    with pytest.raises(RuntimeError, match="worker produced no usable edits"):
        await agent.run()

    assert not list(tmp_path.iterdir()), "a failed worker must not leave files behind"
    assert fake.writes == []
    assert fake.calls, "the model was still asked exactly once"
