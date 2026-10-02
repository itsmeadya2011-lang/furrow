from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.planner import PlannerAgent
from furrow.config import Settings

# The default runner list starts with a bare ``pytest``/``python``, neither of which is
# guaranteed to be on PATH.  Tests pin the candidate list explicitly (usually to
# ``sys.executable -m pytest``) so they exercise the real subprocess path without
# depending on how the test suite itself was invoked.


class StubClient:
    """Minimal stand-in for :class:`furrow.llm.LLMClient` - no network, ever."""

    def __init__(self, responses: list[str], workspace: Path | None = None) -> None:
        self.settings = Settings(workspace=workspace or Path.cwd())
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            return ""
        return self.responses.pop(0)


def _agent(responses: list[str] | None = None, workspace: Path | None = None) -> Any:
    # ``TesterAgent`` is reached through the module so pytest does not try to collect
    # the class itself as a test.
    client = StubClient(responses or [], workspace=workspace)
    return tester_module.TesterAgent(client=cast(Any, client)), client


def _planner(responses: list[str]) -> Any:
    return PlannerAgent(client=cast(Any, StubClient(responses)))


def _use_commands(monkeypatch: pytest.MonkeyPatch, *commands: tuple[str, ...]) -> None:
    monkeypatch.setattr(tester_module, "TEST_COMMANDS", commands)


PYTEST_COMMAND = (sys.executable, "-m", "pytest", "-q")


# --- _run_tests ------------------------------------------------------------


async def test_run_tests_reports_return_code_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "test_fail.py").write_text(
        "def test_intentional_failure():\n    assert False, 'boom'\n", encoding="utf-8"
    )
    _use_commands(monkeypatch, PYTEST_COMMAND)
    agent, _client = _agent(workspace=tmp_path)

    output = await agent._run_tests()

    assert "exit code: 1 (FAILED)" in output
    assert "test_intentional_failure" in output
    assert output.startswith("$ ")


async def test_run_tests_runs_inside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "test_workspace_marker.py").write_text(
        "def test_workspace_marker():\n    assert 0, 'only in the tmp project'\n",
        encoding="utf-8",
    )
    _use_commands(monkeypatch, PYTEST_COMMAND)
    agent, _client = _agent(workspace=tmp_path)

    output = await agent._run_tests()

    assert "test_workspace_marker" in output
    assert "exit code: 1" in output
    # A run rooted at the repository would have collected this suite instead.
    assert "test_run_tests_runs_inside_the_workspace" not in output
    assert "test_llm.py" not in output


async def test_run_tests_passes_cwd_to_the_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_commands(monkeypatch, (sys.executable, "-c", "import os; print(os.getcwd())"))
    agent, _client = _agent(workspace=tmp_path)

    output = await agent._run_tests()

    assert str(tmp_path.resolve()) in output
    assert "exit code: 0 (passed)" in output


async def test_run_tests_falls_through_missing_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_commands(
        monkeypatch,
        ("definitely-not-a-real-runner-abc",),
        ("another-missing-runner-def",),
        (sys.executable, "-c", "print('second runner ran')"),
    )
    agent, _client = _agent()

    output = await agent._run_tests()

    assert "second runner ran" in output
    assert "definitely-not-a-real-runner-abc" not in output
    assert "another-missing-runner-def" not in output
    assert "exit code: 0 (passed)" in output


async def test_run_tests_returns_message_when_nothing_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_commands(monkeypatch, ("nope-not-here-1",), ("nope-not-here-2",))
    agent, _client = _agent()

    assert await agent._run_tests() == tester_module.NO_TEST_RUNNER_MESSAGE


async def test_run_tests_propagates_unexpected_spawn_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("exec exploded")

    monkeypatch.setattr(tester_module.asyncio, "create_subprocess_exec", _boom)
    agent, _client = _agent()

    with pytest.raises(RuntimeError, match="exec exploded"):
        await agent._run_tests()


async def test_run_tests_kills_and_reports_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tester_module, "TEST_TIMEOUT_SECONDS", 0.5)
    _use_commands(
        monkeypatch,
        (sys.executable, "-c", "import time; time.sleep(30)"),
        (sys.executable, "-c", "print('SHOULD NOT RUN')"),
    )
    agent, _client = _agent()

    output = await agent._run_tests()

    assert "timed out after 0.5s" in output
    assert "SHOULD NOT RUN" not in output


# --- TesterAgent.run -------------------------------------------------------


async def test_run_parses_fenced_test_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_commands(monkeypatch, (sys.executable, "-c", "print('1 passed')"))
    agent, client = _agent(
        [
            (
                '```json\n{"passed": false, "summary": "one test broke", '
                '"failures": ["test_intentional_failure"]}\n```'
            )
        ],
        workspace=tmp_path,
    )

    result = await agent.run("do a thing", [])

    assert result.passed is False
    assert result.failures == ["test_intentional_failure"]
    assert "one test broke" in result.summary
    assert "test runner exited with code 0 (passed)" in result.summary
    assert "exit code: 0 (passed)" in client.prompts[0]


async def test_run_surfaces_runner_failure_in_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_commands(monkeypatch, (sys.executable, "-c", "import sys; sys.exit(3)"))
    agent, client = _agent(['{"passed": true, "summary": "looks ok"}'], workspace=tmp_path)

    result = await agent.run("do a thing", [])

    assert "test runner exited with code 3 (FAILED)" in result.summary
    assert "Runner status: test runner exited with code 3 (FAILED)" in client.prompts[0]


async def test_run_falls_back_to_heuristic_on_unparseable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_commands(monkeypatch, (sys.executable, "-c", "print('2 passed')"))
    agent, _client = _agent(["Everything passed, no failures."], workspace=tmp_path)

    result = await agent.run("do a thing", [])

    assert result.passed is True
    assert "Everything passed, no failures." in result.summary
    assert result.failures == []


async def test_run_reports_failure_when_tests_cannot_be_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent, client = _agent()

    async def _boom() -> str:
        raise RuntimeError("no shell")

    monkeypatch.setattr(agent, "_run_tests", _boom)

    result = await agent.run("do a thing", [])

    assert result.passed is False
    assert "no shell" in result.summary
    assert result.failures == ["Failed to run the test suite: no shell"]
    assert client.prompts == []  # never bothered the LLM


# --- PlannerAgent.plan -----------------------------------------------------


async def test_plan_parses_fenced_json_block() -> None:
    planner = _planner(
        [
            (
                "Sure! Here is the plan.\n"
                "```json\n"
                '{"tasks": [{"id": "t1", "description": "write it"}], "rationale": "small"}\n'
                "```\n"
                "Let me know."
            )
        ]
    )

    plan = await planner.plan("ship it")

    assert plan.rationale == "small"
    assert [task.id for task in plan.tasks] == ["t1"]
    assert plan.tasks[0].description == "write it"


async def test_plan_raises_value_error_quoting_unparseable_response() -> None:
    raw = "I'm sorry, I can't produce a plan for that."
    planner = _planner([raw])

    with pytest.raises(ValueError) as excinfo:
        await planner.plan("ship it")

    assert raw in str(excinfo.value)
    assert "Failed to parse plan from LLM" in str(excinfo.value)


async def test_plan_reports_validation_error_for_wrong_shape() -> None:
    raw = '{"tasks": "not a list", "rationale": "oops"}'
    planner = _planner([raw])

    with pytest.raises(ValueError) as excinfo:
        await planner.plan("ship it")

    message = str(excinfo.value)
    assert raw in message
    assert "validation error" in message.lower()
    assert "Plan" in message


async def test_plan_raises_value_error_for_non_object_json() -> None:
    raw = '```json\n[{"id": "t1"}]\n```'
    planner = _planner([raw])

    with pytest.raises(ValueError) as excinfo:
        await planner.plan("ship it")

    assert raw in str(excinfo.value)
    assert "expected a JSON object" in str(excinfo.value)