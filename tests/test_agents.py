"""Unit tests for ``furrow.agents`` -- planner, worker and tester.

These protect the regressions that actually changed observable behaviour:

* ``_normalize_plan`` makes a plan safe for the wave scheduler -- blank
  descriptions are dropped, ids are re-numbered and dependencies remapped,
  dangling dependencies are removed and the plan is capped at
  ``MAX_TASKS`` with a truncation note.
* ``extract_failures`` recognises pytest's ``E`` assertion lines. The old regex
  used ``\\s{3}`` and silently dropped every ``E       assert 1 == 2`` line.
* ``TesterAgent.run`` takes the verdict from the runner's **exit code**; the LLM
  only summarises. A missing test runner is "nothing to verify", not a failure.
* ``_run_tests`` only advances to the next candidate command when the
  executable is missing (``FileNotFoundError``); any other launch error
  surfaces. A timeout kills *and* reaps the child.

Everything is offline: the LLM is a subclass of ``LLMClient`` that answers from
canned text (so the real ``complete_json``/``parse_json_object`` path still runs)
and the subprocess layer is a fake, so no test runner is executed for real.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import furrow.agents.tester as tester_mod
from furrow.agents.planner import MAX_TASKS, PlannerAgent, _normalize_plan
from furrow.agents.tester import (
    MAX_FAILURES,
    MAX_PROMPT_OUTPUT,
    NO_RUNNER_SUMMARY,
    CommandResult,
    extract_failures,
)
from furrow.agents.worker import MAX_REPO_FILES, WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.llm import LLMClient, LLMResponseError

# --------------------------------------------------------------------------- fakes


class FakeLLM(LLMClient):
    """``LLMClient`` double answering from canned text.

    Only ``complete`` is overridden, so ``complete_json`` still runs the real
    ``parse_json_object`` -- a canned reply of ``{"passed": true}`` really is
    parsed, and a canned reply of ``"sorry"`` really does raise.
    """

    def __init__(
        self,
        replies: Sequence[str | Exception],
        settings: Settings,
        files: Sequence[str] | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []
        self.files = list(files) if files is not None else ["furrow/config.py", "furrow/llm.py"]

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def list_files(self, directory: str | Path | None = None) -> list[str]:
        return list(self.files)


class FakeProcess:
    """Subprocess double: returns canned output, or hangs until killed."""

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        exit_code: int = 0,
        hang: bool = False,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = exit_code
        self.hang = hang
        self.returncode: int | None = None if hang else exit_code
        self.killed = False
        self.reaped = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.hang:
            await asyncio.sleep(60)
        self.returncode = self._exit_code
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.reaped = True
        return self.returncode if self.returncode is not None else 0


class Launcher:
    """Records launches and hands out the scripted outcomes in order."""

    def __init__(self, outcomes: Sequence[FakeProcess | BaseException]) -> None:
        self.outcomes = list(outcomes)
        self.launches: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    async def __call__(self, *cmd: str, **kwargs: Any) -> FakeProcess:
        self.launches.append((cmd, kwargs))
        if not self.outcomes:
            raise AssertionError(f"unexpected extra launch: {cmd}")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _install_launcher(monkeypatch: pytest.MonkeyPatch, launcher: Launcher) -> Launcher:
    monkeypatch.setattr(asyncio, "create_subprocess_exec", launcher)
    return launcher


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    """A ``Settings`` copy pointed at ``tmp_path``; the global singleton is never touched."""
    base: dict[str, Any] = {
        "workspace": tmp_path,
        "request_timeout": 1.0,
        "llm_max_retries": 1,
        "anthropic_api_key": None,
        "openai_api_key": None,
        "test_timeout": 0.2,
    }
    base.update(overrides)
    return Settings(_env_file=None).model_copy(update=base)


def _plan_payload(**overrides: Any) -> str:
    payload = {"tasks": [{"id": "1", "description": "do it"}], "rationale": "easy"}
    payload.update(overrides)
    return json.dumps(payload)


# --------------------------------------------------------------------- _normalize_plan


def test_normalize_plan_drops_blank_descriptions_and_renumbers_ids() -> None:
    original = Plan(
        tasks=[
            TaskModel(id="", description="blank id"),
            TaskModel(id="1", description="  spaced  "),
            TaskModel(id="1", description="dup", dependencies=["1", "1", "77"]),
            TaskModel(id="9", description="blank desc", dependencies=["1"]),
            TaskModel(id="4", description="d", dependencies=["1"]),
        ],
        rationale="r",
    )
    got = [(t.id, t.description, t.dependencies) for t in _normalize_plan(original).tasks]
    # A blank *description* is the only drop reason: "blank desc" is a real task and
    # is kept. A duplicate original id ("1") remaps onto the first task that claimed it.
    assert got == [
        ("1", "blank id", []),
        ("2", "spaced", []),
        ("3", "dup", ["2"]),
        ("9", "blank desc", ["2"]),
        ("4", "d", ["2"]),
    ]


def test_normalize_plan_drops_tasks_with_a_blank_description() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="real work"),
            TaskModel(id="2", description="   "),
            TaskModel(id="3", description="\n\t "),
        ],
        rationale="r",
    )
    assert [task.id for task in _normalize_plan(plan).tasks] == ["1"]


def test_normalize_plan_strips_and_renumbers_blank_duplicate_ids() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="  ", description="a"),
            TaskModel(id="7", description="b"),
            TaskModel(id="7", description="c"),
            TaskModel(id="7", description="d"),
        ],
        rationale="r",
    )
    ids = [task.id for task in _normalize_plan(plan).tasks]
    assert ids == ["1", "7", "2", "3"], "ids must be unique, never blank, and renumbered"
    assert len(set(ids)) == len(ids)


def test_normalize_plan_drops_dependencies_on_absent_tasks() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b", dependencies=["1", "77", "does-not-exist"]),
        ],
        rationale="r",
    )
    assert _normalize_plan(plan).tasks[1].dependencies == ["1"]


def test_normalize_plan_drops_dependencies_on_dropped_tasks() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="   "),
            TaskModel(id="2", description="depends on the dropped task", dependencies=["1"]),
        ],
        rationale="r",
    )
    assert _normalize_plan(plan).tasks[0].dependencies == []


def test_normalize_plan_drops_dependencies_truncated_away() -> None:
    plan = Plan(tasks=[TaskModel(id=str(i), description=f"t{i}") for i in range(1, 7)], rationale="r")
    normalised = _normalize_plan(plan)
    assert [task.id for task in normalised.tasks] == ["1", "2", "3", "4", "5"]
    assert all(task.dependencies == [] for task in normalised.tasks)


def test_normalize_plan_caps_at_five_tasks_and_appends_a_truncation_note() -> None:
    plan = Plan(tasks=[TaskModel(id=str(i), description=f"t{i}") for i in range(1, 9)], rationale="why")
    normalised = _normalize_plan(plan)
    assert len(normalised.tasks) == MAX_TASKS == 5
    assert "truncated" in normalised.rationale, normalised.rationale
    assert f"{MAX_TASKS}-task ceiling" in normalised.rationale


def test_normalize_plan_truncation_note_keeps_the_existing_rationale() -> None:
    plan = Plan(tasks=[TaskModel(id=str(i), description=f"t{i}") for i in range(1, 9)], rationale="why")
    rationale = _normalize_plan(plan).rationale
    assert rationale.startswith("why")
    assert "truncated" in rationale


def test_normalize_plan_respects_an_explicit_task_limit() -> None:
    plan = Plan(tasks=[TaskModel(id=str(i), description=f"t{i}") for i in range(1, 5)], rationale="r")
    assert len(_normalize_plan(plan, max_tasks=2).tasks) == 2
    assert len(_normalize_plan(plan, max_tasks=0).tasks) == 0


def test_normalize_plan_deduplicates_repeated_dependencies() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b", dependencies=["1", "1", " 1 "]),
        ],
        rationale="r",
    )
    assert _normalize_plan(plan).tasks[1].dependencies == ["1"]


def test_normalize_plan_leaves_a_valid_plan_unchanged() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="a", description="x", files=["f.py"]),
            TaskModel(id="b", description="y", dependencies=["a"]),
        ],
        rationale="fine",
    )
    normalised = _normalize_plan(plan)
    assert [(t.id, t.description, t.files, t.dependencies) for t in normalised.tasks] == [
        ("a", "x", ["f.py"], []),
        ("b", "y", [], ["a"]),
    ]
    assert normalised.rationale == "fine"


def test_normalize_plan_does_not_mutate_its_input() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="  keep me  ", dependencies=["2"]),
            TaskModel(id="2", description="   "),
        ],
        rationale="r",
    )
    _normalize_plan(plan)
    assert plan.rationale == "r"
    assert [(t.id, t.description, t.dependencies) for t in plan.tasks] == [
        ("1", "  keep me  ", ["2"]),
        ("2", "   ", []),
    ]


def test_normalize_plan_accepts_an_empty_plan() -> None:
    assert _normalize_plan(Plan(tasks=[], rationale="")).tasks == []


# ---------------------------------------------------------------------- extract_failures

_PYTEST_OUTPUT = """$ pytest -q
E       assert 1 == 2
E       assert 1 == 2
E   AssertionError: boom
FAILED tests/test_a.py::test_one - assert 1 == 2
FAILED tests/test_b.py::test_two - x
passed something
"""


def test_extract_failures_captures_pytest_failed_summary_lines() -> None:
    assert extract_failures(_PYTEST_OUTPUT)[2:] == [
        "FAILED tests/test_a.py::test_one",
        "FAILED tests/test_b.py::test_two",
    ]


def test_extract_failures_captures_seven_space_assertion_lines() -> None:
    # Regression: the old ``E\\s{3,}`` regex dropped these because pytest indents
    # the assertion text with 7 spaces.
    assert extract_failures("E       assert 1 == 2\nE       assert 'x' == 'y'\n") == [
        "assert 1 == 2",
        "assert 'x' == 'y'",
    ]


def test_extract_failures_captures_three_space_assertion_lines() -> None:
    assert extract_failures("E   AssertionError: boom\n") == ["AssertionError: boom"]


def test_extract_failures_preserves_output_order() -> None:
    assert extract_failures(_PYTEST_OUTPUT) == [
        "assert 1 == 2",
        "AssertionError: boom",
        "FAILED tests/test_a.py::test_one",
        "FAILED tests/test_b.py::test_two",
    ]


def test_extract_failures_deduplicates_repeated_failures() -> None:
    output = "\n".join(["E       assert 1 == 2"] * 3)
    assert extract_failures(output) == ["assert 1 == 2"]


def test_extract_failures_respects_the_limit_cap() -> None:
    output = "\n".join(f"E   failure number {index}" for index in range(50))
    assert len(extract_failures(output)) == MAX_FAILURES == 20
    assert len(extract_failures(output, limit=3)) == 3


def test_extract_failures_returns_empty_for_clean_output() -> None:
    assert extract_failures("2 passed in 0.01s\n") == []


def test_extract_failures_strips_the_pytest_reason_after_a_dash() -> None:
    assert extract_failures("FAILED tests/test_a.py::test_one - assert 1 == 2\n") == [
        "FAILED tests/test_a.py::test_one"
    ]


# ------------------------------------------------------------------------- PlannerAgent


async def test_planner_parses_json_wrapped_in_a_code_fence(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    fenced = f"Here you go:\n```json\n{_plan_payload()}\n```\nHope that helps"
    client = FakeLLM([fenced], settings)
    plan = await PlannerAgent(client=client, settings=settings).plan("goal")
    assert [task.id for task in plan.tasks] == ["1"]
    assert plan.rationale == "easy"
    assert len(client.calls) == 1, "valid JSON must not trigger the repair prompt"


async def test_planner_parses_json_surrounded_by_prose(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    reply = f"Sure thing, here is the plan: {_plan_payload()} Let me know."
    client = FakeLLM([reply], settings)
    plan = await PlannerAgent(client=client, settings=settings).plan("goal")
    assert [task.description for task in plan.tasks] == ["do it"]


async def test_planner_retries_once_with_a_repair_prompt(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = FakeLLM(["I'm sorry, I cannot help with that.", _plan_payload()], settings)
    plan = await PlannerAgent(client=client, settings=settings).plan("goal")
    assert [task.id for task in plan.tasks] == ["1"]
    assert len(client.calls) == 2, "exactly one repair attempt"
    assert client.calls[0]["prompt"] != client.calls[1]["prompt"]
    assert "code fences" in client.calls[1]["prompt"]
    assert "goal" in client.calls[1]["prompt"], "the repair prompt keeps the original goal"


async def test_planner_raises_after_two_unparseable_attempts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = FakeLLM(["first failure", "second failure"], settings)
    with pytest.raises(LLMResponseError) as excinfo:
        await PlannerAgent(client=client, settings=settings).plan("goal")
    message = str(excinfo.value)
    assert "first failure" in message
    assert "second failure" in message
    assert len(message) < 900, "each excerpt must stay truncated"
    assert len(client.calls) == 2, "no third attempt"


async def test_planner_folds_context_into_the_prompt(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = FakeLLM([_plan_payload()], settings)
    await PlannerAgent(client=client, settings=settings).plan("ship auth", context="cycle 1 findings")
    prompt = client.calls[0]["prompt"]
    assert "ship auth" in prompt
    assert "cycle 1 findings" in prompt


async def test_planner_omits_an_empty_context_section(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = FakeLLM([_plan_payload()], settings)
    await PlannerAgent(client=client, settings=settings).plan("goal", context="   ")
    assert "Context:" not in client.calls[0]["prompt"]


async def test_planner_returns_a_normalised_plan(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    payload = json.dumps(
        {
            "tasks": [
                {"id": "", "description": "first"},
                {"id": "1", "description": "  second  "},
                {"id": "1", "description": "third", "dependencies": ["1"]},
                {"id": "2", "description": "   "},
            ],
            "rationale": "why",
        }
    )
    plan = await PlannerAgent(client=FakeLLM([payload], settings), settings=settings).plan("goal")
    assert [(t.id, t.description, t.dependencies) for t in plan.tasks] == [
        ("1", "first", []),
        ("2", "second", []),
        ("3", "third", ["2"]),
    ]


async def test_planner_uses_the_planner_model(tmp_path: Path) -> None:
    settings = _settings(tmp_path, planner_model="planner-model")
    client = FakeLLM([_plan_payload()], settings)
    await PlannerAgent(client=client, settings=settings).plan("goal")
    assert client.calls[0]["model"] == "planner-model"


async def test_planner_tolerates_a_missing_rationale(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    payload = json.dumps({"tasks": [{"id": "1", "description": "do it"}]})
    plan = await PlannerAgent(client=FakeLLM([payload], settings), settings=settings).plan("goal")
    assert plan.rationale == ""


# -------------------------------------------------------------------------- WorkerAgent


def _worker_client(tmp_path: Path, files: Sequence[str] | None = None) -> FakeLLM:
    return FakeLLM(["worker report"], _settings(tmp_path), files=files)


async def test_worker_returns_the_model_text(tmp_path: Path) -> None:
    client = _worker_client(tmp_path)
    task = TaskModel(id="1", description="Add auth")
    assert await WorkerAgent(task=task, client=client, settings=client.settings).run() == "worker report"


async def test_worker_prompt_carries_id_description_files_and_dependencies(tmp_path: Path) -> None:
    client = _worker_client(tmp_path)
    task = TaskModel(id="3", description="Add auth", files=["a.py", "b.py"], dependencies=["1", "2"])
    await WorkerAgent(task=task, client=client, settings=client.settings).run()
    prompt = client.calls[0]["prompt"]
    assert "Task id: 3" in prompt
    assert "Add auth" in prompt
    assert "a.py, b.py" in prompt
    assert "Depends on: 1, 2" in prompt


async def test_worker_prompt_uses_placeholders_for_missing_files_and_dependencies(
    tmp_path: Path,
) -> None:
    client = _worker_client(tmp_path)
    task = TaskModel(id="1", description="Do something")
    await WorkerAgent(task=task, client=client, settings=client.settings).run()
    prompt = client.calls[0]["prompt"]
    assert "Files to touch: any" in prompt
    assert "Depends on: nothing" in prompt


async def test_worker_passes_the_system_prompt_separately(tmp_path: Path) -> None:
    client = _worker_client(tmp_path)
    task = TaskModel(id="1", description="Do something")
    await WorkerAgent(task=task, client=client, settings=client.settings).run()
    call = client.calls[0]
    assert "worker agent" in call["system"].lower()
    assert "worker agent" not in call["prompt"].lower()


async def test_worker_prompt_lists_the_repository_files(tmp_path: Path) -> None:
    client = _worker_client(tmp_path)
    task = TaskModel(id="1", description="Do something")
    await WorkerAgent(task=task, client=client, settings=client.settings).run()
    prompt = client.calls[0]["prompt"]
    assert "Repository files:" in prompt
    assert "furrow/config.py" in prompt


async def test_worker_prompt_truncates_a_long_repository_file_list(tmp_path: Path) -> None:
    files = [f"pkg/module_{index}.py" for index in range(MAX_REPO_FILES + 300)]
    client = _worker_client(tmp_path, files=files)
    task = TaskModel(id="1", description="Do something")
    await WorkerAgent(task=task, client=client, settings=client.settings).run()
    prompt = client.calls[0]["prompt"]
    assert f"truncated at {MAX_REPO_FILES} files" in prompt
    assert prompt.count(".py") == MAX_REPO_FILES


async def test_worker_uses_the_worker_model(tmp_path: Path) -> None:
    client = FakeLLM(["ok"], _settings(tmp_path, worker_model="worker-model"))
    task = TaskModel(id="1", description="Do something")
    await WorkerAgent(task=task, client=client, settings=client.settings).run()
    assert client.calls[0]["model"] == "worker-model"


# --------------------------------------------------------------------------- TesterAgent


def _tester(
    tmp_path: Path, replies: Sequence[str | Exception], **overrides: Any
) -> tester_mod.TesterAgent:
    settings = _settings(tmp_path, **overrides)
    return tester_mod.TesterAgent(client=FakeLLM(replies, settings), settings=settings)


async def test_tester_verdict_follows_exit_code_not_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    failing = b"E       assert 1 == 2\nFAILED tests/test_a.py::test_one\n"
    agent = _tester(tmp_path, [json.dumps({"passed": True, "summary": "All good", "failures": []})])
    launcher = _install_launcher(
        monkeypatch, Launcher([FakeProcess(stdout=failing, exit_code=1)])
    )
    result = await agent.run("goal", [TaskModel(id="1", description="x")])
    assert result.passed is False, "exit code 1 must win over an optimistic LLM"
    assert "Local verdict" in result.summary
    assert result.failures == ["assert 1 == 2", "FAILED tests/test_a.py::test_one"]
    assert len(launcher.launches) == 1


async def test_tester_passing_exit_code_wins_over_pessimistic_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _tester(
        tmp_path, [json.dumps({"passed": False, "summary": "I think it failed", "failures": ["bogus"]})]
    )
    _install_launcher(monkeypatch, Launcher([FakeProcess(stdout=b"2 passed\n", exit_code=0)]))
    result = await agent.run("goal", [])
    assert result.passed is True, "exit code 0 must win over a pessimistic LLM"
    assert "Local verdict" in result.summary


async def test_tester_prompt_states_the_exit_code_and_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client_replies = [json.dumps({"passed": True, "summary": "ok", "failures": []})]
    agent = _tester(tmp_path, client_replies)
    fake = agent.client
    _install_launcher(
        monkeypatch,
        Launcher([FakeProcess(stdout=b"E   boom\n", stderr=b"warning\n", exit_code=1)]),
    )
    await agent.run("goal", [TaskModel(id="7", description="fix it")])
    prompt = fake.calls[0]["prompt"]
    assert "exit code: 1" in prompt
    assert "authoritative" in prompt
    assert "E   boom" in prompt
    assert "warning" in prompt
    assert "[7] fix it" in prompt
    assert fake.calls[0]["model"] == agent.client.settings.tester_model


async def test_tester_no_runner_is_passed_and_skips_the_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _tester(tmp_path, [])
    fake = agent.client
    monkeypatch.setattr(tester_mod, "TEST_COMMANDS", (("definitely-not-a-real-binary-xyz",),))
    result = await agent.run("goal", [])
    assert result.passed is True
    assert result.summary == NO_RUNNER_SUMMARY
    assert result.failures == []
    assert fake.calls == [], "no runner means nothing to summarise"


async def test_tester_tries_every_candidate_when_none_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tester_mod,
        "TEST_COMMANDS",
        (("definitely-not-a-real-binary-a",), ("definitely-not-a-real-binary-b",)),
    )
    agent = _tester(tmp_path, [])
    result = await agent.run("goal", [])
    assert result.passed is True
    assert result.summary == NO_RUNNER_SUMMARY


async def test_tester_missing_executable_is_the_only_reason_to_try_the_next_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression: `except (FileNotFoundError, Exception)` swallowed every launch error.
    monkeypatch.setattr(
        tester_mod,
        "TEST_COMMANDS",
        (("pytest", "-q"), ("python", "-m", "pytest", "-q")),
    )
    launcher = _install_launcher(monkeypatch, Launcher([PermissionError("not executable")]))
    agent = _tester(tmp_path, [])
    result = await agent.run("goal", [])
    assert result.passed is False
    assert "PermissionError" in result.summary
    assert len(launcher.launches) == 1, "a non-FileNotFoundError must not advance the candidate list"


async def test_tester_llm_failure_falls_back_to_the_local_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _tester(tmp_path, [LLMResponseError("no json here")])
    _install_launcher(
        monkeypatch,
        Launcher([FakeProcess(stdout=b"E       assert 1 == 2\nFAILED t.py::test_a\n", exit_code=1)]),
    )
    result = await agent.run("goal", [])
    assert result.passed is False
    assert "exited with code 1" in result.summary
    assert result.failures == ["assert 1 == 2", "FAILED t.py::test_a"]


async def test_tester_uses_the_failures_reported_by_the_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _tester(
        tmp_path,
        [json.dumps({"passed": False, "summary": "s", "failures": ["from llm", "from llm", "  "]})],
    )
    _install_launcher(monkeypatch, Launcher([FakeProcess(stdout=b"E   local\n", exit_code=1)]))
    result = await agent.run("goal", [])
    assert result.failures == ["from llm", "from llm"], "blank entries are dropped, not deduped"


async def test_tester_caps_the_failures_it_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reported = [f"failure {index}" for index in range(MAX_FAILURES + 10)]
    agent = _tester(tmp_path, [json.dumps({"passed": False, "summary": "s", "failures": reported})])
    _install_launcher(monkeypatch, Launcher([FakeProcess(exit_code=1)]))
    result = await agent.run("goal", [])
    assert len(result.failures) == MAX_FAILURES


async def test_tester_missing_workspace_reports_a_failed_result(tmp_path: Path) -> None:
    agent = _tester(tmp_path, [], workspace=tmp_path / "does-not-exist")
    result = await agent.run("goal", [])
    assert result.passed is False
    assert "FileNotFoundError" in result.summary
    assert result.failures


async def test_tester_prompt_keeps_the_tail_of_long_output(tmp_path: Path) -> None:
    failure = "FAILED tests/test_a.py::test_one\n"
    noise = "x" * (MAX_PROMPT_OUTPUT + 5_000)
    result = CommandResult(ran=True, exit_code=1, output=f"{noise}\n{failure}")
    prompt = tester_mod.TesterAgent._build_prompt("goal", [], result)
    assert failure.strip() in prompt, "the pytest failure summary must survive truncation"
    assert "truncated to the last" in prompt
    assert len(prompt) < len(noise)


# ------------------------------------------------------------------- TesterAgent._run_tests


async def test_run_tests_forwards_the_workspace_as_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path, test_command="pytest -q")
    agent = tester_mod.TesterAgent(client=FakeLLM([], settings), settings=settings)
    launcher = _install_launcher(monkeypatch, Launcher([FakeProcess(exit_code=0)]))
    result = await agent._run_tests()
    assert result.ran is True
    assert result.exit_code == 0
    (_cmd, kwargs) = launcher.launches[0]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["stdout"] is asyncio.subprocess.PIPE
    assert kwargs["stderr"] is asyncio.subprocess.PIPE


async def test_run_tests_splits_the_configured_command_with_shlex(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path, test_command="python3 -m pytest -q -k 'not slow'")
    agent = tester_mod.TesterAgent(client=FakeLLM([], settings), settings=settings)
    launcher = _install_launcher(monkeypatch, Launcher([FakeProcess(exit_code=0)]))
    await agent._run_tests()
    cmd, _kwargs = launcher.launches[0]
    assert cmd == ("python3", "-m", "pytest", "-q", "-k", "not slow")


async def test_run_tests_ignores_a_blank_test_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path, test_command="   ")
    agent = tester_mod.TesterAgent(client=FakeLLM([], settings), settings=settings)
    _install_launcher(monkeypatch, Launcher([FakeProcess(exit_code=0)]))
    result = await agent._run_tests()
    assert result.exit_code == 0, "a blank test_command falls back to the candidate list"


async def test_run_tests_returns_ran_false_when_nothing_launches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tester_mod,
        "TEST_COMMANDS",
        (("definitely-not-a-real-binary-a",), ("definitely-not-a-real-binary-b",)),
    )
    agent = _tester(tmp_path, [])
    result = await agent._run_tests()
    assert result.ran is False
    assert result.exit_code is None
    assert result.output == NO_RUNNER_SUMMARY


async def test_run_tests_timeout_kills_and_reaps_the_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path, test_command="sleep 30", test_timeout=0.05)
    agent = tester_mod.TesterAgent(client=FakeLLM([], settings), settings=settings)
    hung = FakeProcess(hang=True)
    _install_launcher(monkeypatch, Launcher([hung]))
    result = await agent._run_tests()
    assert result.ran is True
    assert result.exit_code is None, "a timeout has no exit code"
    assert "timed out" in result.output
    assert "process killed" in result.output
    assert hung.killed is True, "the timed-out child must be killed"
    assert hung.reaped is True, "the timed-out child must be reaped"


async def test_run_tests_timeout_falls_through_to_the_next_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tester_mod,
        "TEST_COMMANDS",
        (("hangs-forever",), ("pytest", "-q")),
    )
    agent = _tester(tmp_path, [], test_timeout=0.05)
    launcher = _install_launcher(
        monkeypatch, Launcher([FakeProcess(hang=True), FakeProcess(exit_code=0)])
    )
    result = await agent._run_tests()
    assert result.exit_code == 0, "the next candidate ran after the timeout"
    assert len(launcher.launches) == 2


async def test_tester_timeout_result_is_not_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    agent = _tester(
        tmp_path,
        [json.dumps({"passed": True, "summary": "sure, all fine", "failures": []})],
        test_command="sleep 30",
        test_timeout=0.05,
    )
    _install_launcher(monkeypatch, Launcher([FakeProcess(hang=True)]))
    result = await agent.run("goal", [])
    assert result.passed is False, "a timeout is not a pass"
    assert "Local verdict" in result.summary


async def test_run_tests_returns_the_timeout_when_no_candidate_launches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path, test_command="sleep 30", test_timeout=0.05)
    agent = tester_mod.TesterAgent(client=FakeLLM([], settings), settings=settings)
    _install_launcher(monkeypatch, Launcher([FakeProcess(hang=True)]))
    result = await agent._run_tests()
    assert result.ran is True
    assert result.exit_code is None
    assert "timed out" in result.output


async def test_run_tests_stops_at_the_first_candidate_that_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        tester_mod,
        "TEST_COMMANDS",
        (("pytest", "-q"), ("npm", "test")),
    )
    agent = _tester(tmp_path, [])
    launcher = _install_launcher(monkeypatch, Launcher([FakeProcess(exit_code=1)]))
    result = await agent._run_tests()
    assert result.exit_code == 1
    assert len(launcher.launches) == 1, "a non-zero exit code is a real result, not a fallback"


async def test_run_tests_reports_the_command_and_exit_code_in_the_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path, test_command="pytest -q")
    agent = tester_mod.TesterAgent(client=FakeLLM([], settings), settings=settings)
    _install_launcher(
        monkeypatch, Launcher([FakeProcess(stdout=b"1 failed\n", stderr=b"oops\n", exit_code=1)])
    )
    result = await agent._run_tests()
    assert "$ pytest -q" in result.output
    assert "exit code: 1" in result.output
    assert "--- stdout ---\n1 failed" in result.output
    assert "--- stderr ---\noops" in result.output