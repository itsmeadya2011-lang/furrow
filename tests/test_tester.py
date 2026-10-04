"""Tests for :mod:`furrow.agents.tester`: the exit code is the only verdict.

The tester agent measures by running the workspace's real test command and reading its
process result, and explains with an LLM. These tests protect the separation:

* ``passed`` is exactly ``returncode == 0`` and is never inferred from output text -- the
  removed ``"passed" in response.lower()`` heuristic called ``"0 passed, 3 failed"`` a
  win, and the tests below pin the regression shut,
* ``verified is False`` on exactly two paths (no runner detected, or every spawn failed);
  a genuinely failing suite and a timeout are ``verified=True, passed=False``, and
* children are spawned in the workspace with the agent's own credentials scrubbed out.

Nothing here touches the network, and no test spawns a real ``npm``/``cargo``/``go``/
``pytest`` toolchain: the few tests that need real subprocess behaviour use tiny ``sh``
stubs written into a temporary bin directory.
"""

from __future__ import annotations

import os
import shlex
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.tester import (
    CommandResult,
    TesterAgent,
    bounded_output,
    detect_runner,
    format_output,
    run_command,
    tail,
)
from furrow.config import Settings, TaskModel, TestResult
from tests.conftest import FakeLLM

#: ``Callable`` shapes of the shared fixtures in ``tests/conftest.py``.
SettingsFactory = Callable[..., Settings]
FileWriter = Callable[..., Path]

#: What a model returns when it is (wrongly) confident that a red suite is green.
OPTIMISTIC_JSON = '{"passed": true, "summary": "all good", "failures": []}'


# --------------------------------------------------------------------------- #
# stub toolchain helpers
# --------------------------------------------------------------------------- #
def _stub(bin_dir: Path, name: str, body: str) -> None:
    """Write an executable ``/bin/sh`` stub called ``name`` into ``bin_dir``."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / name
    script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    script.chmod(0o755)


def _suite_stub(
    bin_dir: Path,
    *,
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    name: str = "python",
    sleep_for: int | None = None,
) -> None:
    """Write a stub that prints output and exits ``exit_code`` (or hangs forever)."""
    lines: list[str] = []
    if stdout:
        lines.append(f"printf '%s\\n' {shlex.quote(stdout)}")
    if stderr:
        lines.append(f"printf '%s\\n' {shlex.quote(stderr)} >&2")
    if sleep_for is not None:
        # exec so the killed process *is* the sleeper: no orphaned child left behind.
        lines.append(f"exec sleep {sleep_for}")
    else:
        lines.append(f"exit {exit_code}")
    _stub(bin_dir, name, "\n".join(lines))


def _first_on_path(monkeypatch: pytest.MonkeyPatch, bin_dir: Path) -> None:
    """Put ``bin_dir`` ahead of the real PATH so the stub shadows any real toolchain."""
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def _only_on_path(monkeypatch: pytest.MonkeyPatch, bin_dir: Path) -> None:
    """Restrict PATH to ``bin_dir`` so every real tool is simply unavailable."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PATH", str(bin_dir))


def _mark_python(write_file: FileWriter) -> None:
    """Create the Python marker that makes ``detect_runner`` choose ``python -m pytest``."""
    write_file("pyproject.toml", "[project]\nname = 'demo'\nversion = '0.1.0'\n")


def _task(files: Sequence[str] = ()) -> TaskModel:
    return TaskModel(id="1", description="change the greeting", files=list(files))


# --------------------------------------------------------------------------- #
# no runner / no signal
# --------------------------------------------------------------------------- #
async def test_empty_workspace_is_unverified(
    tmp_workspace: Path, make_settings: SettingsFactory
) -> None:
    """No project marker means no runner, so the result carries no verdict at all."""
    fake = FakeLLM([], settings=make_settings(tmp_workspace))

    result = await TesterAgent(client=fake).run("make it work", [])

    assert result.verified is False
    assert result.passed is False
    assert "No test runner detected" in result.summary
    assert result.failures == []
    # Nothing ran, so nothing had to be explained.
    assert fake.calls == 0


async def test_spawn_failure_is_unverified(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A detected runner whose executable cannot be started is unverified, not failed."""
    _mark_python(write_file)
    _only_on_path(monkeypatch, tmp_path / "bin")
    fake = FakeLLM([], settings=make_settings(tmp_workspace))

    result = await TesterAgent(client=fake).run("make it work", [])

    assert result.verified is False
    assert result.passed is False
    assert "No usable test runner" in result.summary
    # Both the primary command and its fallback were attempted and both could not start.
    assert "python -m pytest -q" in result.summary
    assert "pytest -q" in result.summary
    assert fake.calls == 0


# --------------------------------------------------------------------------- #
# the verdict comes from the exit code
# --------------------------------------------------------------------------- #
async def test_detects_python_project_and_passes_on_exit_zero(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``pyproject.toml`` project graded by an exit-0 runner is a verified pass."""
    _mark_python(write_file)
    _suite_stub(tmp_path / "bin", exit_code=0, stdout="1 passed in 0.01s")
    _first_on_path(monkeypatch, tmp_path / "bin")
    fake = FakeLLM(
        ['{"passed": true, "summary": "1 passed", "failures": []}'],
        settings=make_settings(tmp_workspace),
    )

    result = await TesterAgent(client=fake).run("make it work", [])

    runner = detect_runner(tmp_workspace)
    assert runner is not None
    assert runner.ecosystem == "python"
    assert runner.command == ("python", "-m", "pytest", "-q")
    assert runner.marker == "pyproject.toml"
    assert result.verified is True
    assert result.passed is True
    assert result.summary == "1 passed"
    assert result.failures == []


async def test_nonzero_exit_is_a_failure_even_when_the_llm_claims_success(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE regression test: a red suite plus a lying model is still a failure.

    The old implementation graded the model text -- ``passed = "passed" in
    response.lower()`` -- so ``{"passed": true}`` from a confident model turned an
    exit-1 run green. ``passed`` is now ``returncode == 0``, computed before the model is
    consulted and overwritten by nothing; the model's optimism survives only as prose.
    """
    _mark_python(write_file)
    _suite_stub(tmp_path / "bin", exit_code=1, stdout="3 failed", stderr="AssertionError")
    _first_on_path(monkeypatch, tmp_path / "bin")
    fake = FakeLLM([OPTIMISTIC_JSON], settings=make_settings(tmp_workspace))

    result = await TesterAgent(client=fake).run("make it work", [])

    assert result.verified is True
    assert result.passed is False
    # The model's claim is still reported as prose, which is all it is allowed to be.
    assert "all good" in result.summary
    assert any("exited 1" in failure for failure in result.failures)


@pytest.mark.parametrize(
    ("exit_code", "expected_passed"),
    [(1, False), (0, True)],
    ids=["exit-1-reported-failure", "exit-0-reported-pass"],
)
async def test_old_substring_heuristic_would_have_been_wrong(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
    expected_passed: bool,
) -> None:
    """``"3 passed, 1 failed"`` contains "passed": only the exit code can grade it.

    Both runs print the same text and both are graded by the process result, so the
    substring heuristic would have produced the wrong answer for exit 1.
    """
    _mark_python(write_file)
    output = "3 passed, 1 failed"
    _suite_stub(tmp_path / "bin", exit_code=exit_code, stdout=output)
    _first_on_path(monkeypatch, tmp_path / "bin")
    # The old heuristic is what this scripted "model" output encodes.
    fake = FakeLLM(
        [f'{{"passed": true, "summary": "{output}", "failures": []}}'],
        settings=make_settings(tmp_workspace),
    )

    result = await TesterAgent(client=fake).run("make it work", [])

    assert output in fake.prompts[0]
    assert result.verified is True
    assert result.passed is expected_passed
    assert ("passed" in output.lower()) is True  # the substring is present either way


async def test_timeout_is_a_verified_failure(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung runner is killed promptly and graded as a verified failure, not a pass.

    ``DEFAULT_TIMEOUT`` is read at call time, so shortening it here exercises the real
    ``asyncio.wait_for`` + kill/reap path without a two-minute wait.
    """
    _mark_python(write_file)
    _suite_stub(tmp_path / "bin", exit_code=0, sleep_for=120)
    _first_on_path(monkeypatch, tmp_path / "bin")
    monkeypatch.setattr(tester_module, "DEFAULT_TIMEOUT", 0.5)
    fake = FakeLLM([OPTIMISTIC_JSON], settings=make_settings(tmp_workspace))
    agent = TesterAgent(client=fake)

    started = time.monotonic()
    result = await agent.run("make it work", [])
    elapsed = time.monotonic() - started

    assert elapsed < 15
    assert result.verified is True
    assert result.passed is False
    assert any("timed out" in failure.lower() for failure in result.failures)
    assert "TIMED OUT" in fake.prompts[0]


async def test_unparseable_llm_output_still_yields_an_exit_code_verdict(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Useless prose from the model cannot change an exit-0 verdict."""
    _mark_python(write_file)
    _suite_stub(tmp_path / "bin", exit_code=0, stdout="2 passed")
    _first_on_path(monkeypatch, tmp_path / "bin")
    fake = FakeLLM(
        [], default="I am not going to answer with JSON.", settings=make_settings(tmp_workspace)
    )

    result = await TesterAgent(client=fake).run("make it work", [])

    assert result.verified is True
    assert result.passed is True
    assert "PASSED" in result.summary
    assert "LLM summary unavailable" in result.summary
    # The retry budget is bounded: no unbounded prompt loop on a bad model.
    assert fake.calls == tester_module.MAX_ATTEMPTS


async def test_json_fenced_response_is_parsed(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ```` ```json ````-fenced explanation is parsed into the summary and failures."""
    _mark_python(write_file)
    _suite_stub(tmp_path / "bin", exit_code=1, stdout="1 failed")
    _first_on_path(monkeypatch, tmp_path / "bin")
    fenced = (
        "```json\n"
        '{"passed": false, "summary": "one test failed", '
        '"failures": ["tests/test_x.py::test_y - AssertionError"]}\n'
        "```"
    )
    fake = FakeLLM([fenced], settings=make_settings(tmp_workspace))

    result = await TesterAgent(client=fake).run("make it work", [])

    assert result.passed is False
    assert result.summary == "one test failed"
    assert "tests/test_x.py::test_y - AssertionError" in result.failures
    assert fake.calls == 1


# --------------------------------------------------------------------------- #
# the subprocess boundary
# --------------------------------------------------------------------------- #
async def test_environment_is_scrubbed_for_the_subprocess(
    tmp_workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child must not see the keys that pay for the LLM, but keeps its PATH."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "also-secret")
    monkeypatch.setenv("FURROW_WORKER_MODEL", "claude-worker-1")
    _stub(
        tmp_path / "bin",
        "showenv",
        "printf '%s\\n' \"ANTHROPIC=[${ANTHROPIC_API_KEY:-unset}]\" "
        '"OPENAI=[${OPENAI_API_KEY:-unset}]" "FURROW=[${FURROW_WORKER_MODEL:-unset}]" '
        '"PATH_SET=$([ -n "$PATH" ] && echo yes)" "HOME_SET=$([ -n "$HOME" ] && echo yes)"',
    )
    _first_on_path(monkeypatch, tmp_path / "bin")

    outcome = await run_command(("showenv",), cwd=str(tmp_workspace))

    assert outcome.returncode == 0
    assert "ANTHROPIC=[unset]" in outcome.output
    assert "OPENAI=[unset]" in outcome.output
    assert "FURROW=[unset]" in outcome.output
    assert "PATH_SET=yes" in outcome.output
    assert "HOME_SET=yes" in outcome.output
    assert "super-secret" not in outcome.output


async def test_scrubbed_env_drops_credentials_but_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``scrubbed_env()`` removes provider keys and every ``FURROW_*`` setting only."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "also-secret")
    monkeypatch.setenv("FURROW_MODEL", "claude-x")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")

    env = tester_module.scrubbed_env()

    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert not [key for key in env if key.startswith("FURROW_")]
    assert env.get("PATH") == os.environ.get("PATH")


async def test_subprocess_runs_in_the_workspace(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner's cwd is the workspace, never the process cwd (the repo checkout)."""
    _mark_python(write_file)
    _stub(tmp_path / "bin", "python", "printf '%s\\n' \"CWD=$(pwd)\"\nexit 1")
    _first_on_path(monkeypatch, tmp_path / "bin")
    fake = FakeLLM([OPTIMISTIC_JSON], settings=make_settings(tmp_workspace))

    result = await TesterAgent(client=fake).run("make it work", [])

    # Exit 1 keeps the captured output in the measured failures, where it can be read.
    reported = "\n".join(result.failures)
    assert f"CWD={tmp_workspace}" in reported
    assert os.getcwd() not in reported


# --------------------------------------------------------------------------- #
# the judge prompt
# --------------------------------------------------------------------------- #
async def test_tasks_are_passed_to_the_judge_prompt(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explanation prompt lists the files this cycle actually touched."""
    _mark_python(write_file)
    _suite_stub(tmp_path / "bin", exit_code=0, stdout="1 passed")
    _first_on_path(monkeypatch, tmp_path / "bin")
    fake = FakeLLM([OPTIMISTIC_JSON], settings=make_settings(tmp_workspace))
    tasks = [_task(["src/greet.py"]), _task(["README.md"])]

    await TesterAgent(client=fake).run("make it work", tasks)

    prompt = fake.prompts[0]
    assert "Files touched in this cycle:" in prompt
    assert "- src/greet.py" in prompt
    assert "- README.md" in prompt
    assert "make it work" in prompt


async def test_run_lint_and_typecheck_are_opt_in(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lint/type checks stay off by default, and a missing tool never fails the run."""
    _mark_python(write_file)
    _suite_stub(tmp_path / "bin", exit_code=0, stdout="1 passed")
    _only_on_path(monkeypatch, tmp_path / "bin")  # neither ruff nor mypy is reachable
    settings = make_settings(tmp_workspace)
    attempted: list[tuple[str, ...]] = []
    real_run_command = tester_module.run_command

    async def recording_run_command(
        command: tuple[str, ...], *, cwd: str, timeout: float | None = None
    ) -> CommandResult:
        attempted.append(command)
        return await real_run_command(command, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(tester_module, "run_command", recording_run_command)

    default_result = await TesterAgent(client=FakeLLM([OPTIMISTIC_JSON], settings=settings)).run(
        "make it work", []
    )

    assert attempted == [("python", "-m", "pytest", "-q")]
    assert "lint" not in default_result.summary

    attempted.clear()
    opt_in = await TesterAgent(client=FakeLLM([OPTIMISTIC_JSON], settings=settings)).run(
        "make it work", [], run_lint=True
    )

    assert attempted == [
        ("python", "-m", "pytest", "-q"),
        ("ruff", "check", "."),
    ]
    # A missing linter is reported and skipped; it is not a test failure.
    assert opt_in.verified is True
    assert opt_in.passed is True
    assert "unavailable, skipped" in opt_in.summary
    assert opt_in.failures == []


async def test_typecheck_failure_makes_the_run_fail(
    tmp_workspace: Path,
    tmp_path: Path,
    write_file: FileWriter,
    make_settings: SettingsFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``run_typecheck=True`` a failing type check drags the verdict down."""
    _mark_python(write_file)
    _suite_stub(tmp_path / "bin", exit_code=0, stdout="1 passed")
    _stub(tmp_path / "bin", "mypy", "printf '%s\\n' 'error: incompatible type' >&2\nexit 1")
    _only_on_path(monkeypatch, tmp_path / "bin")
    settings = make_settings(tmp_workspace)
    fake = FakeLLM([OPTIMISTIC_JSON], settings=settings)

    result = await TesterAgent(client=fake).run("make it work", [], run_typecheck=True)

    assert result.verified is True
    assert result.passed is False
    assert any("incompatible type" in failure for failure in result.failures)


# --------------------------------------------------------------------------- #
# project detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("lockfile", "expected_command"),
    [
        ("pnpm-lock.yaml", ("pnpm", "test", "--", "--silent")),
        ("yarn.lock", ("yarn", "test", "--silent")),
        (None, ("npm", "test", "--", "--silent")),
    ],
    ids=["pnpm", "yarn", "npm-default"],
)
def test_detects_nodejs_project_via_lockfile(
    tmp_workspace: Path,
    write_file: FileWriter,
    lockfile: str | None,
    expected_command: tuple[str, ...],
) -> None:
    """A ``package.json`` project is graded with the runner its lockfile names."""
    write_file("package.json", '{"name": "demo"}\n')
    if lockfile is not None:
        write_file(lockfile, "lockfileVersion: 9\n")

    runner = detect_runner(tmp_workspace)

    assert runner is not None
    assert runner.ecosystem == "node"
    assert runner.command == expected_command


@pytest.mark.parametrize(
    ("marker", "expected_ecosystem", "expected_command"),
    [
        ("Cargo.toml", "rust", ("cargo", "test", "-q")),
        ("go.mod", "go", ("go", "test", "./...")),
    ],
    ids=["rust", "go"],
)
def test_detects_rust_and_go_projects(
    tmp_workspace: Path,
    write_file: FileWriter,
    marker: str,
    expected_ecosystem: str,
    expected_command: tuple[str, ...],
) -> None:
    """Rust and Go workspaces get their own runner, never a foreign one."""
    write_file(marker, "// marker\n")

    runner = detect_runner(tmp_workspace)

    assert runner is not None
    assert runner.ecosystem == expected_ecosystem
    assert runner.command == expected_command
    assert runner.marker == marker


def test_python_marker_wins_over_a_foreign_marker(
    tmp_workspace: Path, write_file: FileWriter
) -> None:
    """Python is detected first, so a stray ``go.mod`` cannot hijack a Python repo."""
    _mark_python(write_file)
    write_file("go.mod", "module demo\n")

    runner = detect_runner(tmp_workspace)

    assert runner is not None
    assert runner.ecosystem == "python"


# --------------------------------------------------------------------------- #
# pure output helpers
# --------------------------------------------------------------------------- #
def test_bounded_output_truncates_and_keeps_the_tail() -> None:
    """Over-long output is clamped to the tail, with a note about what was dropped."""
    text = "HEADMARKER" + ("x" * (2 * tester_module.MAX_OUTPUT_CHARS)) + "TAILMARKER"

    bounded = bounded_output(text)

    assert len(bounded) < len(text)
    assert bounded.endswith("TAILMARKER")
    assert "HEADMARKER" not in bounded
    assert "earlier characters truncated" in bounded


def test_bounded_output_reports_blank_output() -> None:
    """Empty output is labelled rather than passed through as an empty string."""
    assert bounded_output("   \n  ") == "(no output)"


def test_format_output_labels_streams_separately() -> None:
    """stdout and stderr stay distinguishable instead of being glued together."""
    formatted = format_output("all good\n", "warning: slow\n")

    assert "--- stdout ---\nall good" in formatted
    assert "--- stderr ---\nwarning: slow" in formatted


def test_tail_keeps_the_end_of_the_output() -> None:
    """The informative tail (where test runners print reasons) survives truncation."""
    text = "noise\n" * 500 + "tests/test_x.py::test_y FAILED"

    assert tail(text, limit=40).endswith("tests/test_x.py::test_y FAILED")
    assert tail("short", limit=40) == "short"


def test_command_result_reports_its_command_line() -> None:
    """``CommandResult.command_line`` renders the command for humans and for prompts."""
    outcome = CommandResult(("python", "-m", "pytest", "-q"), 1, "boom")

    assert outcome.command_line == "python -m pytest -q"
    assert outcome.timed_out is False
    assert outcome.spawn_failed is False


def test_test_result_is_not_collected_as_a_test_case() -> None:
    """``TestResult`` opts out of pytest collection despite its name."""
    result = TestResult(passed=True, summary="ok", verified=True)

    assert TestResult.__test__ is False
    assert result.passed is True
