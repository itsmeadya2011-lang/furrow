"""Regression tests for :class:`furrow.agents.tester.TesterAgent`.

These run a *real* ``pytest`` subprocess, so they are kept few and cheap: the
temporary project holds two test functions and nothing else.  ``pytest`` is only
available as a module in this sandbox, so ``conftest.write_pytest_shim`` puts a
tiny executable on ``PATH`` for the tester to find; nothing is installed into
the repository.

Load-bearing regressions pinned here:

* a failing run's output is *not* discarded (the exit-code-ignoring bug);
* the non-JSON reply heuristic never reports ``passed=True`` from weak evidence;
* a huge output keeps its tail and says that it was truncated;
* a nonexistent workspace yields a clean :class:`TestResult`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from furrow.agents.tester import TEST_OUTPUT_TAIL_LIMIT as OUTPUT_TAIL_LIMIT
from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient
from tests.conftest import (
    TESTER,
    ScriptedCompletion,
    executable_shim,
    install_scripted,
    make_settings,
    prepend_to_path,
    verdict_json,
)

PASSING_SUITE = "def test_one():\n    assert 1 + 1 == 2\n\n\ndef test_two():\n    assert True\n"
FAILING_SUITE = "def test_three():\n    assert 1 + 1 == 3\n"


def make_project(root: Path, body: str = PASSING_SUITE, *, config: str = "pyproject.toml") -> Path:
    """A minimal pytest project inside *root*."""
    project = root / "proj"
    project.mkdir(parents=True)
    if config:
        (project / config).write_text("[project]\nname = 'sample'\nversion = '0'\n")
    (project / "test_sample.py").write_text(body)
    return project


def make_tester(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, reply: str | Any
) -> tuple[TesterAgent, ScriptedCompletion]:
    scripted = install_scripted(monkeypatch, tester=reply)
    return TesterAgent(client=LLMClient(settings=settings)), scripted


# --------------------------------------------------------- real pytest runs


async def test_a_passing_project_reports_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, available_pytest_cli: Path
) -> None:
    project = make_project(tmp_path)
    tester, scripted = make_tester(
        monkeypatch, make_settings(project), verdict_json(True, "2 passed")
    )
    result = await tester.run("Goal: everything passes", [])
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.failures == []
    # The real pytest output must have reached the model.
    assert "2 passed" in scripted.only_prompt(TESTER)


async def test_a_failing_project_reports_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, available_pytest_cli: Path
) -> None:
    project = make_project(tmp_path, PASSING_SUITE + FAILING_SUITE)
    tester, scripted = make_tester(
        monkeypatch,
        make_settings(project),
        verdict_json(False, "1 failed", ["test_three failed: assert 1 + 1 == 3"]),
    )
    result = await tester.run("Goal: everything passes", [])
    assert result.passed is False
    assert result.failures
    assert "test_three" in result.failures[0]
    output = scripted.only_prompt(TESTER)
    assert "1 failed" in output or "failed" in output


async def test_the_goal_reaches_the_tester_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, available_pytest_cli: Path
) -> None:
    project = make_project(tmp_path)
    tester, scripted = make_tester(monkeypatch, make_settings(project), verdict_json(True, "ok"))
    await tester.run("Goal: make the widget blue", [])
    assert "make the widget blue" in scripted.only_prompt(TESTER)


async def test_uses_the_tester_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, available_pytest_cli: Path
) -> None:
    project = make_project(tmp_path)
    settings = make_settings(project, tester_model="tester-model-x")
    tester, scripted = make_tester(monkeypatch, settings, verdict_json(True, "ok"))
    await tester.run("Goal", [])
    assert scripted.models(TESTER) == ["tester-model-x"]


async def test_a_failing_run_output_is_not_discarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, available_pytest_cli: Path
) -> None:
    """The exit code must not hide the evidence the model needs."""
    project = make_project(tmp_path, PASSING_SUITE + FAILING_SUITE)
    tester, scripted = make_tester(
        monkeypatch, make_settings(project), verdict_json(False, "1 failed", ["boom"])
    )
    await tester.run("Goal", [])
    assert "test_three" in scripted.only_prompt(TESTER)


async def test_a_named_tasks_list_does_not_change_the_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, available_pytest_cli: Path
) -> None:
    project = make_project(tmp_path)
    tester, _ = make_tester(monkeypatch, make_settings(project), verdict_json(True, "ok"))
    tasks = [TaskModel(id="1", description="d")]
    result = await tester.run("Goal", tasks)
    assert result.passed is True


# ------------------------------------------------------- test_command override


async def test_the_test_command_override_is_honoured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = make_project(tmp_path)
    marker = f"CUSTOM-RUNNER-{tmp_path.name}"
    settings = make_settings(project, test_command=f"{sys.executable} -c \"print('{marker}')\"")
    tester, scripted = make_tester(monkeypatch, settings, verdict_json(True, "ok"))
    result = await tester.run("Goal", [])
    assert result.passed is True
    assert marker in scripted.only_prompt(TESTER)


async def test_the_test_command_override_replaces_auto_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, available_pytest_cli: Path
) -> None:
    """With ``test_command`` set, the discovered pytest run must not happen."""
    project = make_project(tmp_path, PASSING_SUITE + FAILING_SUITE)
    settings = make_settings(project, test_command=f"{sys.executable} -c \"print('ONLY THIS')\"")
    tester, scripted = make_tester(monkeypatch, settings, verdict_json(True, "ok"))
    await tester.run("Goal", [])
    output = scripted.only_prompt(TESTER)
    assert "ONLY THIS" in output
    assert "test_three" not in output


async def test_a_failing_test_command_still_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = make_project(tmp_path)
    settings = make_settings(
        project, test_command=f"{sys.executable} -c \"import sys; print('BADNEWS'); sys.exit(3)\""
    )
    tester, scripted = make_tester(monkeypatch, settings, verdict_json(False, "failed", ["bad"]))
    result = await tester.run("Goal", [])
    assert result.passed is False
    assert "BADNEWS" in scripted.only_prompt(TESTER)


async def test_a_timing_out_command_is_reported_not_hung(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The runner is killed at ``test_timeout`` and the suite is told, not hung."""
    project = make_project(tmp_path)
    # ``exec`` keeps the shell from forking, so the timed-out child is our own and
    # is reaped instead of lingering as an orphan.
    settings = make_settings(
        project,
        test_command=f"exec {sys.executable} -c 'import time; time.sleep(30)'",
        test_timeout=1,
    )
    tester, scripted = make_tester(monkeypatch, settings, verdict_json(False, "timeout", ["slow"]))
    result = await run_with_deadline(tester.run("Goal", []))
    assert result.passed is False
    assert "timed out after 1s" in scripted.only_prompt(TESTER)


async def test_a_timing_out_auto_detected_runner_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same guarantee for the auto-detected (non-shell) runner."""
    project = make_project(tmp_path)
    bindir = tmp_path / "bin"
    executable_shim(bindir, "pytest", "import time\ntime.sleep(30)\n")
    prepend_to_path(monkeypatch, bindir)
    tester, scripted = make_tester(
        monkeypatch, make_settings(project, test_timeout=1), verdict_json(False, "timeout", ["s"])
    )
    result = await run_with_deadline(tester.run("Goal", []))
    assert result.passed is False
    assert "timed out after 1s" in scripted.only_prompt(TESTER)


async def test_a_command_that_does_not_exist_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = make_project(tmp_path)
    settings = make_settings(project, test_command="definitely-not-a-real-command-xyz")
    tester, scripted = make_tester(monkeypatch, settings, verdict_json(False, "cannot run", ["x"]))
    result = await tester.run("Goal", [])
    assert result.passed is False
    assert "definitely-not-a-real-command-xyz" in scripted.only_prompt(TESTER)


async def run_with_deadline(coro: Any, timeout: float = 25.0) -> Any:
    """Await *coro* with a hard deadline so a hung runner cannot hang the suite."""
    return await asyncio.wait_for(coro, timeout=timeout)


# ------------------------------------------------------------------- Node-only


async def test_a_node_only_project_does_not_look_like_a_python_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A Node project must not hand a Node crash back as the test result."""
    project = tmp_path / "node-proj"
    project.mkdir()
    (project / "package.json").write_text('{"name": "node-proj", "version": "1.0.0"}')
    bindir = tmp_path / "bin"
    executable_shim(
        bindir,
        "npm",
        "import sys\n"
        "sys.stdout.write('> node-proj test\\n')\n"
        "sys.stderr.write('npm ERR! missing script: test\\n')\n"
        "sys.stderr.write('Error: Cannot find module \\'./index.js\\'\\n')\n"
        "sys.exit(1)\n",
    )
    prepend_to_path(monkeypatch, bindir)

    settings = make_settings(project)
    tester, scripted = make_tester(
        monkeypatch, settings, verdict_json(False, "the runner could not start", ["npm"])
    )
    result = await tester.run("Goal", [])
    assert result.passed is False
    assert result.failures == ["npm"]
    assert result.summary == "the runner could not start"


async def test_a_node_only_project_output_is_not_judged_a_pass_by_the_heuristic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With the raw runner text as the reply, the heuristic must not say "pass"."""
    project = tmp_path / "node-proj"
    project.mkdir()
    (project / "package.json").write_text('{"name": "node-proj", "version": "1.0.0"}')
    bindir = tmp_path / "bin"
    executable_shim(
        bindir,
        "npm",
        "import sys\n"
        "sys.stderr.write('npm ERR! code ENOENT\\n')\n"
        "sys.stderr.write('Error: Cannot find module \\'./index.js\\'\\n')\n"
        "sys.exit(1)\n",
    )
    prepend_to_path(monkeypatch, bindir)

    seen: list[str] = []

    async def echo_the_output(call: Any) -> str:
        seen.append(call.prompt)
        return call.prompt.split("Test output:", 1)[-1].strip()

    install_scripted(monkeypatch, tester=echo_the_output)
    settings = make_settings(project)
    result = await TesterAgent(client=LLMClient(settings=settings)).run("Goal", [])
    assert result.passed is False
    assert result.failures
    assert "Cannot find module" in seen[0]


async def test_no_runner_at_all_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty project with no runner on PATH must not be reported as passing."""
    project = tmp_path / "empty"
    project.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    tester, scripted = make_tester(
        monkeypatch, make_settings(project), verdict_json(False, "no", ["x"])
    )
    result = await tester.run("Goal", [])
    assert result.passed is False
    assert "No test runner found" in scripted.only_prompt(TESTER)


# ---------------------------------------------------------------- truncation


async def test_a_huge_output_keeps_its_tail_and_notes_truncation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = make_project(tmp_path)
    marker = f"TAIL-MARKER-{tmp_path.name}"
    filler = "F" * 500
    settings = make_settings(
        project,
        test_command=(f"{sys.executable} -c \"print('{filler}' * 200); print('{marker}')\""),
    )
    tester, scripted = make_tester(monkeypatch, settings, verdict_json(False, "long", ["x"]))
    result = await run_with_deadline(tester.run("Goal", []))
    assert result.passed is False
    prompt = scripted.only_prompt(TESTER)
    assert marker in prompt
    assert "test output truncated to last 15000 chars" in prompt
    assert "Test output:\n...\n" in prompt  # the head was dropped
    assert len(prompt) < OUTPUT_TAIL_LIMIT + 4000  # the tail budget is respected


async def test_a_small_output_is_not_marked_as_truncated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = make_project(tmp_path)
    settings = make_settings(project, test_command=f"{sys.executable} -c \"print('SHORT')\"")
    tester, scripted = make_tester(monkeypatch, settings, verdict_json(True, "ok"))
    await tester.run("Goal", [])
    assert "truncated" not in scripted.only_prompt(TESTER)
    assert "SHORT" in scripted.only_prompt(TESTER)


# ------------------------------------------------------------ the heuristics


@pytest.mark.parametrize(
    "reply",
    [
        "not passed",
        "The suite did not pass.",
        "2 passed, 1 failed",
        "one test failed",
        "assertion failed in test_three",
        "There was an error while collecting.",
        "everything looks fine",
        "",
    ],
)
async def test_a_non_json_reply_never_reports_success_from_weak_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reply: str
) -> None:
    project = make_project(tmp_path)
    settings = make_settings(project, test_command=f'{sys.executable} -c "pass"')
    tester, _ = make_tester(monkeypatch, settings, reply)
    result = await tester.run("Goal", [])
    assert result.passed is False


async def test_a_non_json_all_passed_reply_does_report_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = make_project(tmp_path)
    settings = make_settings(project, test_command=f'{sys.executable} -c "pass"')
    tester, _ = make_tester(monkeypatch, settings, "I ran them: all passed")
    result = await tester.run("Goal", [])
    assert result.passed is True
    assert result.failures == []


async def test_a_malformed_verdict_falls_back_to_the_heuristic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A verdict without ``summary`` cannot be built, so the heuristic takes over."""
    project = make_project(tmp_path)
    settings = make_settings(project, test_command=f'{sys.executable} -c "pass"')
    tester, _ = make_tester(monkeypatch, settings, json.dumps({"passed": True}))
    result = await tester.run("Goal", [])
    assert result.passed is False


async def test_the_json_verdict_is_used_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = make_project(tmp_path)
    settings = make_settings(project, test_command=f'{sys.executable} -c "pass"')
    tester, _ = make_tester(
        monkeypatch,
        settings,
        verdict_json(True, "5 passed in 0.1s", ["ignored on purpose"]),
    )
    result = await tester.run("Goal", [])
    assert result.passed is True
    assert result.summary == "5 passed in 0.1s"
    assert result.failures == ["ignored on purpose"]


# --------------------------------------------------------------- workspace


async def test_a_nonexistent_workspace_yields_a_clean_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "never-created"
    tester, scripted = make_tester(
        monkeypatch, make_settings(missing), verdict_json(False, "no ws", ["x"])
    )
    result = await tester.run("Goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is False
    assert not missing.exists(), "the tester must not create a missing workspace"
    assert "Workspace does not exist" in scripted.only_prompt(TESTER)


async def test_a_tester_without_a_client_uses_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from furrow.config import settings as global_settings

    install_scripted(monkeypatch, tester=verdict_json(True, "ok"))
    tester = TesterAgent()
    assert tester.client.settings is global_settings
