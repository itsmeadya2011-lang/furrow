"""Regression tests for ``furrow.agents.tester``.

Every test here is hermetic. ``TesterAgent._run_tests`` discovers candidates via
``shutil.which`` and then spawns them through ``asyncio.create_subprocess_exec``,
so both seams are replaced: no real process is ever started. That matters here
specifically because the bare ``pytest -q`` candidate is the first entry in the
runner table and ``pytest`` really is on PATH, so an unpatched run would execute
this very test suite recursively.

Settings are duck-typed with ``types.SimpleNamespace`` on purpose: the tester only
reads ``workspace``, ``tester_model`` and ``test_timeout``, and the tests must stay
correct whether or not ``Settings.test_timeout`` happens to exist right now.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import tester as tester_mod

# Aliased on import: names beginning with ``Test`` make pytest try to collect them.
from furrow.agents.tester import (
    DEFAULT_TEST_TIMEOUT,
    MAX_TEST_OUTPUT_CHARS,
    TEST_OUTPUT_HEAD_CHARS,
)
from furrow.agents.tester import (
    TesterAgent as Tester,
)
from furrow.config import Settings, TaskModel
from furrow.config import TestResult as Verdict

# Tools the runner table probes for. ``python`` is absent on PATH in this
# environment, which is why the "python -m pytest" candidate needs an explicit
# positive case before it can ever be selected.
ALL_TOOLS = ("pytest", "python", "python3", "npm", "pnpm", "yarn", "cargo", "go")

# (tool, project marker file, expected argv) for each non-python candidate.
ECOSYSTEMS = (
    ("npm", "package.json", ["npm", "test", "--", "--silent"]),
    ("pnpm", "package.json", ["pnpm", "test", "--", "--silent"]),
    ("yarn", "package.json", ["yarn", "test", "--silent"]),
    ("cargo", "Cargo.toml", ["cargo", "test", "-q"]),
    ("go", "go.mod", ["go", "test", "./..."]),
)


# --------------------------------------------------------------------------- stubs


def which_for(*available: str):
    """A ``shutil.which`` replacement that only "finds" the named executables."""

    allowed = set(available)

    def fake_which(cmd: str, *args: Any, **kwargs: Any) -> str | None:
        return f"/usr/local/bin/{cmd}" if cmd in allowed else None

    return fake_which


class FakeProcess:
    """Minimal stand-in for ``asyncio.subprocess.Process``."""

    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int | None = 0,
        hang: bool = False,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            # Never completes: only asyncio.wait_for can end this coroutine.
            await asyncio.Event().wait()
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int | None:
        self.waited = True
        return self.returncode


class SpawnRecorder:
    """Replaces ``asyncio.create_subprocess_exec`` and records every spawn."""

    def __init__(self, result: FakeProcess | BaseException) -> None:
        self._result = result
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> FakeProcess:
        self.calls.append((args, kwargs))
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result

    @property
    def commands(self) -> list[list[str]]:
        return [list(args) for args, _ in self.calls]


class StubLLM:
    """LLM client double; records prompts and returns a canned JSON verdict."""

    def __init__(self, settings: Any, reply: str) -> None:
        self.settings = settings
        self.reply = reply
        self.prompts: list[str] = []
        self.models: list[str | None] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        self.models.append(model)
        return self.reply


def make_settings(workspace: Path, **overrides: Any) -> types.SimpleNamespace:
    values: dict[str, Any] = {
        "workspace": workspace,
        "test_timeout": DEFAULT_TEST_TIMEOUT,
        "tester_model": "stub-tester-model",
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def install_spawn(monkeypatch: pytest.MonkeyPatch, result: FakeProcess | BaseException) -> SpawnRecorder:
    recorder = SpawnRecorder(result)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", recorder)
    return recorder


def install_which(monkeypatch: pytest.MonkeyPatch, *available: str) -> None:
    monkeypatch.setattr(tester_mod.shutil, "which", which_for(*available))


def make_agent(tmp_path: Path, **overrides: Any) -> Tester:
    return Tester(settings=make_settings(tmp_path, **overrides))


def force_timeout(
    monkeypatch: pytest.MonkeyPatch, effective: float = 0.01
) -> list[float | None]:
    """Record the timeout handed to ``asyncio.wait_for`` and force it to fire fast.

    Returning the list lets a test assert that the *configured* seconds reached the
    subprocess layer without the test itself waiting for them.
    """
    real_wait_for = asyncio.wait_for
    seen: list[float | None] = []

    def spy(awaitable: Any, timeout: float | None = None) -> Any:
        seen.append(timeout)
        return real_wait_for(awaitable, effective)

    monkeypatch.setattr(asyncio, "wait_for", spy)
    return seen


# ------------------------------------------------- 1. availability gating (the bug)


async def test_candidate_with_missing_executable_is_skipped(tmp_path, monkeypatch):
    """A candidate whose executable is absent is skipped, never spawned."""
    proc = FakeProcess(returncode=0)
    spawn = install_spawn(monkeypatch, proc)
    install_which(monkeypatch)  # nothing is installed

    report = await make_agent(tmp_path)._run_tests()

    assert spawn.calls == []
    assert report.startswith("No test runner found.")
    assert proc.killed is False


@pytest.mark.parametrize(("tool", "marker", "argv"), ECOSYSTEMS)
async def test_runner_needs_project_marker_file(tmp_path, monkeypatch, tool, marker, argv):
    """A tool binary alone is not enough: the project must look like that ecosystem.

    This is the fixed bug - the old code ran the first *spawnable* candidate, so a
    Python-only workspace with npm on PATH would report npm's "no test script"
    noise instead of skipping it.
    """
    spawn = install_spawn(monkeypatch, FakeProcess(returncode=0))
    install_which(monkeypatch, tool)  # the binary "exists"

    report = await make_agent(tmp_path)._run_tests()

    assert spawn.commands == [], f"{tool} ran without {marker}: {report}"
    assert "No test runner found." in report
    assert f"{tool} test" in report

    # Positive control: with the marker file present the very same tool is chosen.
    (tmp_path / marker).write_text("{}\n", encoding="utf-8")
    report = await make_agent(tmp_path)._run_tests()

    assert spawn.commands == [argv]
    assert f"$ {' '.join(argv)} (exit 0)" in report


async def test_python_pytest_runner_needs_a_pytest_project(tmp_path, monkeypatch):
    """``python -m pytest`` requires both an interpreter and evidence of a test suite."""
    spawn = install_spawn(monkeypatch, FakeProcess(returncode=0))
    install_which(monkeypatch, "python3")  # interpreter present, but no tests/ dir

    report = await make_agent(tmp_path)._run_tests()
    assert spawn.commands == []
    assert "No test runner found." in report

    (tmp_path / "tests").mkdir()
    report = await make_agent(tmp_path)._run_tests()

    assert len(spawn.commands) == 1
    assert spawn.commands[0][1:] == ["-m", "pytest", "-q"]
    assert "--- stdout ---" in report


# ------------------------------------- 2. a non-zero exit is a failure, not a skip


async def test_nonzero_exit_is_reported_not_skipped(tmp_path, monkeypatch):
    """A runner that executes and fails is reported; the next candidate never runs."""
    spawn = install_spawn(
        monkeypatch,
        FakeProcess(
            stdout=b"2 failed, 1 passed in 0.12s",
            stderr=b"E   assert 1 == 2",
            returncode=1,
        ),
    )
    install_which(monkeypatch, "pytest", "python3", "npm", "pnpm", "yarn", "cargo", "go")

    report = await make_agent(tmp_path)._run_tests()

    assert spawn.commands == [["pytest", "-q"]], "fell through to another runner after failure"
    assert "(exit 1)" in report
    assert "2 failed, 1 passed" in report
    assert "assert 1 == 2" in report
    for name in ("npm", "cargo", "go test"):
        assert name not in report, f"{name} should never have been considered"


# ------------------------------------------------------- 3. report formatting


async def test_report_names_runner_and_labels_streams(tmp_path, monkeypatch):
    install_spawn(
        monkeypatch,
        FakeProcess(stdout=b"  1 passed  \n", stderr=b"", returncode=0),
    )
    install_which(monkeypatch, "pytest")

    report = await make_agent(tmp_path)._run_tests()

    assert report.startswith("$ pytest -q (exit 0)")
    assert "--- stdout ---" in report
    assert "--- stderr ---" in report
    assert "1 passed" in report
    # stdout and stderr are separated, not concatenated.
    assert report.index("--- stdout ---") < report.index("--- stderr ---")


async def test_empty_streams_get_a_placeholder(tmp_path, monkeypatch):
    install_spawn(monkeypatch, FakeProcess(stdout=b"", stderr=b"", returncode=0))
    install_which(monkeypatch, "pytest")

    report = await make_agent(tmp_path)._run_tests()

    assert report.count("(empty)") == 2
    assert "--- stdout ---\n(empty)" in report
    assert "--- stderr ---\n(empty)" in report


# ------------------------------------------------------------------- 4. cwd


async def test_subprocess_runs_in_the_configured_workspace(tmp_path, monkeypatch):
    spawn = install_spawn(monkeypatch, FakeProcess(returncode=0))
    install_which(monkeypatch, "pytest")

    workspace = tmp_path / "project"
    workspace.mkdir()
    agent = Tester(settings=make_settings(workspace))
    report = await agent._run_tests()

    assert len(spawn.calls) == 1
    _args, kwargs = spawn.calls[0]
    assert kwargs["cwd"] == str(workspace)
    assert Path(kwargs["cwd"]).resolve() == workspace.resolve()
    assert kwargs["env"] is None  # inherit, not an empty environment
    assert "--- stdout ---" in report


# ---------------------------------------------------------------- 5. timeouts


async def test_timeout_reports_configured_seconds_and_kills_the_process(tmp_path, monkeypatch):
    proc = FakeProcess(hang=True)
    install_spawn(monkeypatch, proc)
    install_which(monkeypatch, "pytest")
    seen = force_timeout(monkeypatch)

    report = await make_agent(tmp_path, test_timeout=2)._run_tests()

    assert seen[0] == 2.0, "configured test_timeout never reached asyncio.wait_for"
    assert "timed out after 2s" in report
    assert "pytest -q" in report
    assert proc.killed is True
    assert proc.waited is True


@pytest.mark.parametrize("bad_timeout", [0, -1, 0.0, -30, "abc", None, "not-a-number"])
async def test_invalid_test_timeout_falls_back_to_default(tmp_path, monkeypatch, bad_timeout):
    install_spawn(monkeypatch, FakeProcess(stdout=b"ok", returncode=0))
    install_which(monkeypatch, "pytest")
    seen = force_timeout(monkeypatch)

    agent = make_agent(tmp_path, test_timeout=bad_timeout)
    assert agent._timeout() == DEFAULT_TEST_TIMEOUT

    report = await agent._run_tests()

    assert seen[0] == DEFAULT_TEST_TIMEOUT
    assert "timed out" not in report
    assert "--- stdout ---" in report


def test_valid_test_timeout_is_used_verbatim(tmp_path, monkeypatch):
    install_which(monkeypatch)
    agent = make_agent(tmp_path, test_timeout=45)
    assert agent._timeout() == 45.0
    agent = make_agent(tmp_path, test_timeout="90")
    assert agent._timeout() == 90.0


# -------------------------------------------------- 6. Settings.test_timeout contract


def test_settings_test_timeout_default_and_env(monkeypatch):
    """``test_timeout`` defaults to 300 and honours ``FURROW_TEST_TIMEOUT``.

    Tolerant of the field not existing yet: ``Settings`` stores pydantic fields in
    ``model_fields`` rather than on the class, so the guard checks the mapping.
    """
    fields = getattr(Settings, "model_fields", {})
    if "test_timeout" not in fields:
        pytest.skip("Settings.test_timeout is not defined in this checkout")

    monkeypatch.delenv("FURROW_TEST_TIMEOUT", raising=False)
    assert fields["test_timeout"].default == 300
    assert Settings(_env_file=None).test_timeout == 300

    monkeypatch.setenv("FURROW_TEST_TIMEOUT", "42")
    assert Settings(_env_file=None).test_timeout == 42


# ------------------------------------------------------------- 7. truncation


async def test_huge_output_is_truncated_with_an_explicit_marker(tmp_path, monkeypatch):
    huge = "H" * (MAX_TEST_OUTPUT_CHARS * 3)
    install_spawn(monkeypatch, FakeProcess(stdout=huge.encode(), stderr=b"warn", returncode=2))
    install_which(monkeypatch, "pytest")

    report = await make_agent(tmp_path)._run_tests()

    assert "--- stdout ---" in report
    assert "--- stderr ---" in report
    assert "characters truncated" in report
    # head + tail are kept, the middle is dropped
    assert report.count("H") == TEST_OUTPUT_HEAD_CHARS + MAX_TEST_OUTPUT_CHARS
    assert "warn" in report
    assert len(report) < len(huge) + 500


async def test_short_output_is_not_truncated(tmp_path, monkeypatch):
    install_spawn(monkeypatch, FakeProcess(stdout=b"all good", stderr=b"", returncode=0))
    install_which(monkeypatch, "pytest")

    report = await make_agent(tmp_path)._run_tests()

    assert "characters truncated" not in report
    assert "all good" in report


# ------------------------------------------------ 8. no runner available message


async def test_no_runner_message_lists_every_checked_candidate(tmp_path, monkeypatch):
    spawn = install_spawn(monkeypatch, FakeProcess(returncode=0))
    install_which(monkeypatch)

    report = await make_agent(tmp_path)._run_tests()

    assert "No test runner found." in report
    assert "Checked:" in report
    checked = report.split("Checked:\n", 1)[1]
    assert len(tester_mod._RUNNERS) == len(checked.strip().splitlines())
    for runner in tester_mod._RUNNERS:
        assert f"  - {runner.name} (not available)" in checked
    assert spawn.calls == []


async def test_missing_workspace_is_reported_without_spawning(tmp_path, monkeypatch):
    spawn = install_spawn(monkeypatch, FakeProcess(returncode=0))
    install_which(monkeypatch, "pytest", "npm")

    missing = tmp_path / "does-not-exist"
    agent = Tester(settings=make_settings(missing))
    report = await agent._run_tests()

    assert spawn.calls == []
    assert "No test runner found." in report
    assert "workspace does not exist" in report


# ------------------------------------------------------------- 9. run() contract


async def test_run_returns_a_test_result(tmp_path, monkeypatch):
    install_which(monkeypatch)
    agent_client = StubLLM(
        make_settings(tmp_path),
        '{"passed": true, "summary": "All tests passed", "failures": []}',
    )
    agent = Tester(client=agent_client)

    result = await agent.run("ship the feature", [TaskModel(id="1", description="do it")])

    assert isinstance(result, Verdict)
    assert result.passed is True
    assert result.failures == []
    assert agent_client.models == ["stub-tester-model"]
    prompt = agent_client.prompts[0]
    assert "ship the feature" in prompt
    assert "No test runner found." in prompt  # the real output reached the prompt


async def test_run_reports_execution_errors_instead_of_raising(tmp_path, monkeypatch):
    install_which(monkeypatch, "pytest")
    install_spawn(monkeypatch, RuntimeError("spawn exploded"))
    agent = Tester(client=StubLLM(make_settings(tmp_path), '{"passed": true}'))

    result = await agent.run("goal", [])

    assert isinstance(result, Verdict)
    assert result.passed is False
    assert any("spawn exploded" in failure for failure in result.failures)
    assert "spawn exploded" in result.summary


async def test_run_falls_back_to_substring_verdict_on_unparseable_reply(tmp_path, monkeypatch):
    install_which(monkeypatch)
    agent = Tester(client=StubLLM(make_settings(tmp_path), "Looks like the suite PASSED to me"))

    result = await agent.run("goal", [])

    assert result.passed is True
    assert result.summary == "Looks like the suite PASSED to me"