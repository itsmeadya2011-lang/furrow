"""Tests for :mod:`furrow.agents`: parsing, planner, worker and tester.

All three agents are driven through :class:`FakeLLMClient`, which inherits the
production sandboxed filesystem helpers, so a worker really writes files into
``tmp_path`` and a tester really spawns a subprocess. Nothing contacts a
provider.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

import furrow.agents.tester as tester_module
from furrow.agents.parsing import (
    coerce_status,
    extract_json,
    parse_json_response,
    strip_code_fence,
)
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import CommandResult
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TaskStatus, TestResult

from .conftest import FakeLLMClient

# --------------------------------------------------------------------------- #
# strip_code_fence
# --------------------------------------------------------------------------- #


def test_strip_code_fence_removes_a_json_fence() -> None:
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_removes_a_bare_fence() -> None:
    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_code_fence_leaves_unfenced_text_alone() -> None:
    assert strip_code_fence('  {"a": 1}  ') == '{"a": 1}'


def test_strip_code_fence_unwraps_a_fence_wrapped_in_blank_lines() -> None:
    assert strip_code_fence('```json\n\n{"a": 1}\n\n```') == '{"a": 1}'


def test_strip_code_fence_keeps_the_payload_of_a_fence_then_prose() -> None:
    """Trailing prose defeats fence stripping, but ``extract_json`` still copes."""
    text = '```json\n{"a": 1}\n```\nthanks!'

    assert "```json" not in strip_code_fence(text)
    assert json.loads(extract_json(text)) == {"a": 1}


# --------------------------------------------------------------------------- #
# extract_json
# --------------------------------------------------------------------------- #


def test_extract_json_skips_leading_prose() -> None:
    assert extract_json('Sure! Here is the plan:\n{"tasks": []}') == '{"tasks": []}'


def test_extract_json_skips_trailing_commentary() -> None:
    text = '{"tasks": []}\n\nLet me know if you want changes!'

    assert extract_json(text) == '{"tasks": []}'


def test_extract_json_handles_a_fenced_object_with_prose() -> None:
    text = 'Here you go:\n```json\n{"a": {"b": 2}}\n```\nDone.'

    assert json.loads(extract_json(text)) == {"a": {"b": 2}}


def test_extract_json_raises_without_braces() -> None:
    with pytest.raises(ValueError, match="no"):
        extract_json("there is no json here")


def test_extract_json_raises_without_a_closing_brace() -> None:
    with pytest.raises(ValueError, match="no closing"):
        extract_json('{"a": 1')


# --------------------------------------------------------------------------- #
# parse_json_response
# --------------------------------------------------------------------------- #


def test_parse_json_response_accepts_bare_json() -> None:
    plan = parse_json_response('{"tasks": [], "rationale": "r"}', Plan, "planner response")

    assert plan.rationale == "r"


def test_parse_json_response_accepts_a_fenced_chatty_reply() -> None:
    text = 'Sure thing!\n```json\n{"tasks": [{"id": "1", "description": "d"}]}\n```\nEnjoy.'

    plan = parse_json_response(text, Plan, "planner response")

    assert plan.tasks[0].id == "1"


def test_parse_json_response_normalises_status_keys() -> None:
    text = '{"tasks": [{"id": "1", "description": "d", "status": "in progress"}]}'

    plan = parse_json_response(text, Plan, "planner response")

    assert plan.tasks[0].status is TaskStatus.RUNNING


def test_parse_json_response_raises_on_malformed_json() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_json_response("{not json}", Plan, "planner response")

    assert "planner response" in str(excinfo.value)


def test_parse_json_response_raises_on_a_payload_that_fails_validation() -> None:
    with pytest.raises(ValueError, match="planner response"):
        parse_json_response('{"tasks": "not a list"}', Plan, "planner response")


def test_parse_json_response_raises_on_a_non_object_payload() -> None:
    """A bare JSON array is rejected before it is even decoded."""
    with pytest.raises(ValueError) as excinfo:
        parse_json_response("[1, 2, 3]", Plan, "planner response")

    assert "planner response" in str(excinfo.value)


def test_parse_json_response_error_mentions_the_context() -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_json_response("nothing", TestResult, "tester response")

    assert "tester response" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# coerce_status
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pending", TaskStatus.PENDING),
        ("running", TaskStatus.RUNNING),
        ("completed", TaskStatus.COMPLETED),
        ("failed", TaskStatus.FAILED),
        ("skipped", TaskStatus.SKIPPED),
    ],
)
def test_coerce_status_accepts_the_canonical_values(raw: str, expected: TaskStatus) -> None:
    assert coerce_status(raw) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("done", TaskStatus.COMPLETED),
        ("complete", TaskStatus.COMPLETED),
        ("DONE", TaskStatus.COMPLETED),
        ("  success  ", TaskStatus.COMPLETED),
        ("in progress", TaskStatus.RUNNING),
        ("in-progress", TaskStatus.RUNNING),
        ("in_progress", TaskStatus.RUNNING),
        ("not_started", TaskStatus.PENDING),
        ("not started", TaskStatus.PENDING),
        ("todo", TaskStatus.PENDING),
        ("blocked", TaskStatus.SKIPPED),
        ("cancelled", TaskStatus.SKIPPED),
        ("errored", TaskStatus.FAILED),
    ],
)
def test_coerce_status_absorbs_aliases(raw: str, expected: TaskStatus) -> None:
    assert coerce_status(raw) is expected


def test_coerce_status_defaults_to_pending() -> None:
    assert coerce_status("who knows") is TaskStatus.PENDING
    assert coerce_status(None) is TaskStatus.PENDING
    assert coerce_status(17) is TaskStatus.PENDING
    assert coerce_status(object()) is TaskStatus.PENDING


def test_coerce_status_passes_through_an_enum_member() -> None:
    assert coerce_status(TaskStatus.FAILED) is TaskStatus.FAILED


# --------------------------------------------------------------------------- #
# planner
# --------------------------------------------------------------------------- #


def _plan_payload(count: int, dependencies: dict[str, list[str]] | None = None) -> str:
    """Build a raw planner reply with ``count`` tasks.

    Args:
        count: How many tasks to emit.
        dependencies: Optional per-task dependency lists.

    Returns:
        A chatty, fenced JSON reply.
    """
    tasks = []
    for index in range(1, count + 1):
        task: dict[str, Any] = {
            "id": str(index),
            "description": f"task {index}",
            "files": [f"pkg/task{index}.py"],
            "dependencies": (dependencies or {}).get(str(index), []),
        }
        tasks.append(task)
    body = json.dumps({"tasks": tasks, "rationale": "because"})
    return f"Sure, here is the plan:\n```json\n{body}\n```\nLet me know!"


async def test_planner_parses_a_chatty_fenced_reply(settings_factory: Any) -> None:
    settings = settings_factory(planner_model="planner-model")
    client = FakeLLMClient(settings=settings, responses=[_plan_payload(2)])

    plan = await PlannerAgent(client=client).plan("build a thing")

    assert [task.id for task in plan.tasks] == ["1", "2"]
    assert client.models == ["planner-model"]
    assert "build a thing" in client.prompts[0]


async def test_planner_repairs_one_malformed_reply(settings_factory: Any) -> None:
    client = FakeLLMClient(settings=settings_factory(), responses=["garbage", _plan_payload(1)])

    plan = await PlannerAgent(client=client).plan("build a thing")

    assert len(plan.tasks) == 1
    assert len(client.calls) == 2
    assert "rejected" in client.prompts[1]
    assert "build a thing" in client.prompts[1]


async def test_planner_raises_after_two_malformed_replies(settings_factory: Any) -> None:
    client = FakeLLMClient(settings=settings_factory(), responses=["garbage", "also garbage"])

    with pytest.raises(ValueError, match="repair"):
        await PlannerAgent(client=client).plan("build a thing")

    assert len(client.calls) == 2


async def test_planner_truncates_an_oversized_plan(settings_factory: Any) -> None:
    client = FakeLLMClient(settings=settings_factory(), responses=[_plan_payload(7)])

    plan = await PlannerAgent(client=client).plan("build a thing")

    assert len(plan.tasks) == 5
    assert "Truncated 2 task(s)" in plan.rationale
    assert "because" in plan.rationale


async def test_planner_drops_unknown_and_self_referential_dependencies(
    settings_factory: Any,
) -> None:
    payload = _plan_payload(
        2, {"1": ["2", "ghost"], "2": ["2", "ghost"]}
    )
    client = FakeLLMClient(settings=settings_factory(), responses=[payload])

    plan = await PlannerAgent(client=client).plan("build a thing")

    by_id = {task.id: task for task in plan.tasks}
    assert by_id["1"].dependencies == ["2"]
    assert by_id["2"].dependencies == []
    assert "unknown or self-referential" in plan.rationale


async def test_planner_rejects_an_empty_task_list(settings_factory: Any) -> None:
    body = json.dumps({"tasks": [], "rationale": "nothing to do"})
    client = FakeLLMClient(settings=settings_factory(), responses=[body])

    with pytest.raises(ValueError, match="no tasks"):
        await PlannerAgent(client=client).plan("build a thing")


async def test_planner_normalises_planned_statuses(settings_factory: Any) -> None:
    body = json.dumps(
        {
            "tasks": [
                {"id": "1", "description": "d", "status": "done"},
                {"id": "2", "description": "d", "status": "in progress"},
            ]
        }
    )
    client = FakeLLMClient(settings=settings_factory(), responses=[body])

    plan = await PlannerAgent(client=client).plan("build a thing")

    assert [task.status for task in plan.tasks] == [TaskStatus.COMPLETED, TaskStatus.RUNNING]


async def test_planner_uses_its_own_settings_without_a_client(settings_factory: Any) -> None:
    settings = settings_factory(planner_model="explicit-planner")

    planner = PlannerAgent(settings=settings)

    assert planner.settings is settings


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #


def _worker_payload(summary: str, files: dict[str, str]) -> str:
    """Build a raw worker reply.

    Args:
        summary: The summary line.
        files: Mapping of path to content.

    Returns:
        A fenced JSON reply.
    """
    body = json.dumps(
        {
            "summary": summary,
            "files": [{"path": path, "content": content} for path, content in files.items()],
        }
    )
    return f"```json\n{body}\n```"


async def test_worker_writes_files_and_returns_the_summary(
    settings_factory: Any, tmp_path: Path
) -> None:
    settings = settings_factory(worker_model="worker-model")
    task = TaskModel(id="7", description="add the module", files=["pkg/new.py"])
    reply = _worker_payload("Added pkg/new.py.", {"pkg/new.py": "x = 1\n"})
    client = FakeLLMClient(settings=settings, responses=[reply])

    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()

    assert summary == "Added pkg/new.py."
    assert worker.files_written == ["pkg/new.py"]
    assert worker.errors == []
    assert (tmp_path / "pkg" / "new.py").read_text(encoding="utf-8") == "x = 1\n"
    assert client.models == ["worker-model"]


async def test_worker_malformed_reply_degrades_without_raising(settings_factory: Any) -> None:
    task = TaskModel(id="1", description="d")
    client = FakeLLMClient(settings=settings_factory(), responses=["  I could not comply.  "])

    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()

    assert summary == "I could not comply."
    assert worker.files_written == []
    assert len(worker.errors) == 1
    assert "worker response" in worker.errors[0]


async def test_worker_records_a_sandbox_escape_but_still_writes_the_good_file(
    settings_factory: Any, tmp_path: Path
) -> None:
    task = TaskModel(id="1", description="d")
    reply = _worker_payload(
        "wrote two files", {"good.py": "ok\n", "../escaped.py": "nope\n"}
    )
    client = FakeLLMClient(settings=settings_factory(), responses=[reply])

    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()

    assert summary == "wrote two files"
    assert worker.files_written == ["good.py"]
    assert len(worker.errors) == 1
    assert "escaped.py" in worker.errors[0]
    assert (tmp_path / "good.py").read_text(encoding="utf-8") == "ok\n"
    assert not (tmp_path.parent / "escaped.py").exists()


async def test_worker_falls_back_to_the_raw_text_for_an_empty_summary(
    settings_factory: Any,
) -> None:
    task = TaskModel(id="1", description="d")
    reply = _worker_payload("", {"a.py": "x\n"})
    client = FakeLLMClient(settings=settings_factory(), responses=[reply])

    assert await WorkerAgent(task=task, client=client).run() == reply.strip()


def test_worker_prompt_carries_the_task_identity(settings_factory: Any) -> None:
    settings = settings_factory()
    (settings.workspace / "pkg").mkdir(parents=True, exist_ok=True)
    (settings.workspace / "pkg" / "mod.py").write_text("existing = True\n", encoding="utf-8")
    task = TaskModel(
        id="42",
        description="rewrite the parser",
        files=["pkg/mod.py"],
        dependencies=["41"],
    )
    client = FakeLLMClient(settings=settings)

    prompt = WorkerAgent(task=task, client=client).build_prompt()

    assert "Task id: 42" in prompt
    assert "Task description: rewrite the parser" in prompt
    assert "pkg/mod.py" in prompt
    assert "Dependencies: 41" in prompt
    assert "existing = True" in prompt


def test_worker_prompt_includes_a_previous_result(settings_factory: Any) -> None:
    task = TaskModel(id="1", description="d", result="attempt 1 wrote nothing")
    client = FakeLLMClient(settings=settings_factory())

    assert "attempt 1 wrote nothing" in WorkerAgent(task=task, client=client).build_prompt()


async def test_worker_propagates_llm_errors(settings_factory: Any) -> None:
    from furrow.llm import LLMError

    task = TaskModel(id="1", description="d")
    client = FakeLLMClient(
        settings=settings_factory(), responses=[LLMError("provider exploded")]
    )

    with pytest.raises(LLMError, match="provider exploded"):
        await WorkerAgent(task=task, client=client).run()


async def test_worker_reports_a_fully_escaping_reply(settings_factory: Any) -> None:
    task = TaskModel(id="1", description="d")
    reply = _worker_payload("nothing written", {"../../escaped.py": "nope\n"})
    client = FakeLLMClient(settings=settings_factory(), responses=[reply])

    worker = WorkerAgent(task=task, client=client)
    await worker.run()

    assert worker.files_written == []
    assert len(worker.errors) == 1


# --------------------------------------------------------------------------- #
# tester: command detection
# --------------------------------------------------------------------------- #


def test_tester_honours_the_explicit_test_command(settings_factory: Any) -> None:
    settings = settings_factory(test_command="uv run pytest -q")

    detected = tester_module.TesterAgent(settings=settings).detect_command()

    assert detected == ["uv", "run", "pytest", "-q"]


def test_tester_probes_known_runners_in_order(
    settings_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tester_module.shutil, "which", lambda name: f"/usr/bin/{name}")
    settings = settings_factory(test_command=None)

    assert tester_module.TesterAgent(settings=settings).detect_command() == ["pytest", "-q"]


def test_tester_skips_runners_that_are_not_installed(
    settings_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        tester_module.shutil,
        "which",
        lambda name: "/usr/bin/npm" if name == "npm" else None,
    )
    settings = settings_factory(test_command=None)

    detected = tester_module.TesterAgent(settings=settings).detect_command()

    assert detected == ["npm", "test", "--silent"]


def test_tester_detects_no_runner(
    settings_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tester_module.shutil, "which", lambda name: None)
    settings = settings_factory(test_command=None)

    assert tester_module.TesterAgent(settings=settings).detect_command() == []


# --------------------------------------------------------------------------- #
# tester: CommandResult.ok truth table
# --------------------------------------------------------------------------- #


def test_command_result_ok_for_a_clean_exit() -> None:
    assert CommandResult(command=["pytest"], returncode=0, output="").ok is True


def test_command_result_not_ok_for_a_non_zero_exit() -> None:
    assert CommandResult(command=["pytest"], returncode=1, output="boom").ok is False


def test_command_result_not_ok_when_timed_out() -> None:
    result = CommandResult(
        command=["pytest"], returncode=0, output="", timed_out=True, error="killed"
    )

    assert result.ok is False


def test_command_result_not_ok_when_the_runner_errored() -> None:
    result = CommandResult(command=["pytest"], returncode=0, output="", error="no such binary")

    assert result.ok is False


def test_command_result_not_ok_without_a_returncode() -> None:
    assert CommandResult(command=["pytest"], returncode=None, output="").ok is False


# --------------------------------------------------------------------------- #
# tester: run()
# --------------------------------------------------------------------------- #


def _exit_command(status: int = 0) -> str:
    """Build a command string that exits with ``status``.

    Args:
        status: The exit status.

    Returns:
        A ``sys.executable`` one-liner.
    """
    return f"{sys.executable} -c \"import sys; sys.exit({status})\""


async def test_tester_reports_a_pass_for_a_clean_exit(settings_factory: Any) -> None:
    settings = settings_factory(test_command=_exit_command(0))
    client = FakeLLMClient(
        settings=settings, responses=[json.dumps({"passed": True, "summary": "all good"})]
    )

    result = await tester_module.TesterAgent(client=client).run("goal")

    assert result.passed is True
    assert result.summary == "all good"


async def test_tester_downgrades_a_false_pass_verdict(settings_factory: Any) -> None:
    settings = settings_factory(test_command=_exit_command(3))
    client = FakeLLMClient(
        settings=settings, responses=[json.dumps({"passed": True, "summary": "looks fine"})]
    )

    result = await tester_module.TesterAgent(client=client).run("goal")

    assert result.passed is False
    assert "overruled" in result.summary
    assert "status 3" in result.summary
    assert any("status 3" in failure for failure in result.failures)


async def test_tester_downgrades_a_false_pass_after_a_timeout(settings_factory: Any) -> None:
    script = "import time; time.sleep(30)"
    settings = settings_factory(test_command=f"{sys.executable} -c \"{script}\"", test_timeout=0.3)
    client = FakeLLMClient(settings=settings, responses=[json.dumps({"passed": True})])

    result = await tester_module.TesterAgent(client=client).run("goal")

    assert result.passed is False
    assert "overruled" in result.summary
    assert result.failures and "test_timeout" in result.failures[0]


async def test_tester_falls_back_to_the_exit_status_when_the_verdict_is_unusable(
    settings_factory: Any,
) -> None:
    settings = settings_factory(test_command=_exit_command(1))
    client = FakeLLMClient(settings=settings, responses=["I have no opinion"])

    result = await tester_module.TesterAgent(client=client).run("goal")

    assert result.passed is False
    assert "not valid JSON" in result.summary
    assert any("status 1" in failure for failure in result.failures)


async def test_tester_fallback_reports_success_when_the_command_passed(
    settings_factory: Any,
) -> None:
    settings = settings_factory(test_command=_exit_command(0))
    client = FakeLLMClient(settings=settings, responses=["I have no opinion"])

    result = await tester_module.TesterAgent(client=client).run("goal")

    assert result.passed is True
    assert result.failures == []


async def test_tester_reports_no_runner_found(
    settings_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tester_module.shutil, "which", lambda name: None)
    settings = settings_factory(test_command=None)
    client = FakeLLMClient(settings=settings)

    result = await tester_module.TesterAgent(client=client).run("goal")

    assert result.passed is False
    assert result.summary == "No test runner found"
    assert result.failures and "test_command" in result.failures[0]
    assert client.calls == []


async def test_tester_prompt_includes_the_goal_tasks_and_output(
    settings_factory: Any,
) -> None:
    settings = settings_factory(test_command=_exit_command(1), tester_model="tester-model")
    client = FakeLLMClient(settings=settings, responses=[json.dumps({"passed": False})])
    tasks = [
        TaskModel(id="1", description="add a health endpoint", status=TaskStatus.COMPLETED)
    ]

    await tester_module.TesterAgent(client=client).run("ship the health endpoint", tasks)

    prompt = client.prompts[0]
    assert "ship the health endpoint" in prompt
    assert "add a health endpoint" in prompt
    assert "id=1" in prompt
    assert "exit status 1" in prompt
    assert client.models == ["tester-model"]


async def test_tester_prompt_survives_no_tasks(settings_factory: Any) -> None:
    settings = settings_factory(test_command=_exit_command(0))
    client = FakeLLMClient(settings=settings, responses=[json.dumps({"passed": True})])

    await tester_module.TesterAgent(client=client).run("goal", None)

    assert "(no task records were provided)" in client.prompts[0]


# --------------------------------------------------------------------------- #
# tester: run_command
# --------------------------------------------------------------------------- #


async def test_run_command_defaults_cwd_to_the_workspace(settings_factory: Any) -> None:
    settings = settings_factory()

    result = await tester_module.TesterAgent(settings=settings).run_command(
        [sys.executable, "-c", "import os; print(os.getcwd())"]
    )

    assert result.ok is True
    assert Path(result.output.strip()).resolve() == Path(settings.workspace).resolve()


async def test_run_command_honours_an_explicit_cwd(
    settings_factory: Any, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    other.mkdir()

    result = await tester_module.TesterAgent(settings=settings_factory()).run_command(
        [sys.executable, "-c", "import os; print(os.getcwd())"], cwd=other
    )

    assert Path(result.output.strip()).resolve() == other.resolve()


async def test_run_command_captures_a_non_zero_exit(settings_factory: Any) -> None:
    result = await tester_module.TesterAgent(settings=settings_factory()).run_command(
        [sys.executable, "-c", "import sys; sys.stderr.write('bad\\n'); sys.exit(2)"]
    )

    assert result.returncode == 2
    assert result.ok is False
    assert "bad" in result.output


async def test_run_command_reports_a_missing_binary(settings_factory: Any) -> None:
    result = await tester_module.TesterAgent(settings=settings_factory()).run_command(
        ["definitely-not-a-real-binary-xyz"]
    )

    assert result.returncode is None
    assert result.ok is False
    assert result.error and "could not run" in result.error


async def test_run_command_with_an_empty_command(settings_factory: Any) -> None:
    result = await tester_module.TesterAgent(settings=settings_factory()).run_command([])

    assert result.ok is False
    assert result.error == "No test command to run"


async def test_run_command_timeout_kills_the_process_and_keeps_partial_output(
    settings_factory: Any,
) -> None:
    settings = settings_factory(test_timeout=0.5)
    script = "import sys, time; print('partial output', flush=True); time.sleep(30)"

    tester = tester_module.TesterAgent(settings=settings)

    result = await tester.run_command([sys.executable, "-c", script])

    assert result.timed_out is True
    assert result.ok is False
    assert "partial output" in result.output
    assert result.error and "test_timeout" in result.error


def test_tester_explicit_settings_win_over_the_clients(settings_factory: Any) -> None:
    """The implementation prefers the ``settings`` argument, contradicting the
    agent docstrings, which claim the client's settings win. See the bug report:
    ``furrow/agents/tester.py:107`` (and the identical ``settings or (...)``
    lines in ``planner.py:45`` and ``worker.py:100``)."""
    from_client = settings_factory(test_command="from-client")
    from_argument = settings_factory(test_command="from-argument")

    tester = tester_module.TesterAgent(
        client=FakeLLMClient(settings=from_client), settings=from_argument
    )

    assert tester.settings is from_argument
    assert tester.detect_command() == ["from-argument"]


def test_tester_takes_the_client_settings_when_no_settings_are_given(
    settings_factory: Any,
) -> None:
    from_client = settings_factory(test_command="from-client")

    tester = tester_module.TesterAgent(client=FakeLLMClient(settings=from_client))

    assert tester.settings is from_client
    assert tester.detect_command() == ["from-client"]


def test_tester_result_is_not_collected_as_a_test() -> None:
    assert TestResult.__test__ is False
