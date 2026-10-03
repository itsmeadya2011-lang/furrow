"""Hermetic tester tests: no network, no API keys, no 120s waits.

Three seams keep everything fast and deterministic:

* ``shutil.which`` is monkeypatched (``./.venv/bin`` is deliberately not on PATH,
  so an in-process probe would otherwise disagree with the shell), and
  :func:`furrow.agents.tester.detect_test_commands` also accepts the probe
  directly.
* ``tester.run_command`` is replaced by a :class:`Recorder`, so the agent never
  spawns the project's own suite. The one real subprocess used (the
  killpg/zombie test) sleeps 60s in a child that is killed after 0.25s.
* ``LLMClient.complete`` is replaced by :class:`FakeClient`, which replays canned
  strings and records prompts.

The tests therefore assert the *tester's own contract*: what counts as evidence,
what counts as "nothing to verify", and which signal wins.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.prompts import TESTER_PROMPT
from furrow.agents.tester import (
    DEFAULT_TEST_TIMEOUT,
    MAX_OUTPUT_CHARS,
    MAX_PROMPT_TASKS,
    NO_PARSE_SUMMARY,
    NO_RUNNER_SUMMARY,
    NO_TEST_RUNNER_KIND,
    NO_TESTS_COLLECTED_EXIT,
    TEST_PHASE_BUDGET_FACTOR,
    TEST_RESULT_KIND_ERROR,
    TEST_RESULT_KIND_RAN,
    TRUNCATION_MARKER,
    CommandOutcome,
    TesterAgent,
    VerificationReport,
    build_env,
    detect_static_checks,
    detect_test_commands,
    extract_json_object,
    heuristic_passed,
    parse_llm_result,
    run_command,
)
from furrow.config import Settings, TaskModel, TestResult

# Imported for typing/assertions, not for collection.
TesterAgent.__test__ = False
TestResult.__test__ = False

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

#: Vocabulary that would make the orchestrator (or the planner that reads the
#: summary) believe there is something to fix.
FAILURE_WORDS = ("fail", "error", "broken", "traceback", "fix", "must")


def make_settings(**overrides: Any) -> Settings:
    """Settings isolated from the environment so a stray .env cannot change a test."""
    overrides.setdefault("workspace", Path("/nonexistent/workspace"))
    return Settings(_env_file=None, **overrides)


def fake_which(*available: str):
    """A ``shutil.which`` stand-in: resolved for ``available``, ``None`` otherwise."""

    def which(name: str) -> str | None:
        return f"/usr/local/bin/{name}" if name in set(available) else None

    return which


def always_true(_name: str) -> bool:
    return True


def always_false(_name: str) -> bool:
    return False


def marker_exists(*names: str):
    """An ``exists`` stand-in matching only the given file/dir names."""
    wanted = set(names)

    def exists(path: Path) -> bool:
        return path.name in wanted

    return exists


def outcome_for(argv: list[str], **fields: Any) -> CommandOutcome:
    """A :class:`CommandOutcome` shaped like a real one, with short labels."""
    return CommandOutcome(
        argv=list(argv),
        label=fields.pop("label", tester_module._label_for(argv)),
        timeout=fields.pop("timeout", 120.0),
        **fields,
    )


def verdict_json(passed: bool, summary: str = "s", failures: list[str] | None = None) -> str:
    return json.dumps({"passed": passed, "summary": summary, "failures": failures or []})


class FakeClient:
    """Stands in for LLMClient: canned replies, recorded prompts, recorded model."""

    def __init__(self, responses: list[str] | None = None, settings: Settings | None = None) -> None:
        self.responses = list(responses or [verdict_json(True, "ok")])
        self.prompts: list[str] = []
        self.models: list[str | None] = []
        self.settings = settings or make_settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        self.models.append(model)
        return self.responses[min(len(self.prompts) - 1, len(self.responses) - 1)]


class SpyClient(FakeClient):
    """A client whose ``complete`` always raises, to test provider outages."""

    def __init__(self, error: Exception, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.raise_error = error

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append(prompt)
        raise self.raise_error


class FakeClock:
    """Monotonic clock the test advances by hand, so budgets are exercised instantly."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Recorder:
    """Replacement for ``tester.run_command``: records calls, replays outcomes."""

    def __init__(self, outcomes: dict[str, dict[str, Any]] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.calls: list[dict[str, Any]] = []
        self.on_call = None

    async def __call__(self, argv: list[str], **kwargs: Any) -> CommandOutcome:
        self.calls.append({"argv": list(argv), **kwargs})
        if self.on_call is not None:
            self.on_call(argv, kwargs)
        return outcome_for(list(argv), **self.outcomes.get(argv[0], {"exit_code": 0}))

    @property
    def argvs(self) -> list[list[str]]:
        return [call["argv"] for call in self.calls]


class FakeProcess:
    """Minimal stand-in for ``asyncio.subprocess.Process``.

    ``pid`` is a *nonexistent* pid, so ``os.killpg`` raises ``ProcessLookupError``
    and the ``proc.kill()`` fallback is what runs; the real killpg path is covered
    by the real-subprocess test.
    """

    pid = 2**22 - 1

    def __init__(
        self,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        hang: bool = False,
        wait_hangs: bool = False,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._hang = hang
        self.wait_hangs = wait_hangs
        self.returncode: int | None = None
        self.kill_calls = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(3600)
        self.returncode = self._returncode
        return self._stdout, self._stderr

    async def wait(self) -> int:
        if self.wait_hangs:
            await asyncio.sleep(3600)
        self.returncode = self._returncode
        return self._returncode

    def kill(self) -> None:
        self.kill_calls += 1


async def spawned(proc: FakeProcess) -> FakeProcess:
    """``create_subprocess_exec`` replacement that hands back a prepared process."""
    return proc


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Install a :class:`Recorder` as ``tester.run_command`` so nothing executes."""
    recorder = Recorder()
    monkeypatch.setattr(tester_module, "run_command", recorder)
    return recorder


def make_agent(
    monkeypatch: pytest.MonkeyPatch,
    *,
    binaries: tuple[str, ...] = ("pytest",),
    markers: tuple[str, ...] = ("pyproject.toml",),
    modules: Any = always_false,
    responses: list[str] | None = None,
    client: FakeClient | None = None,
    **setting_overrides: Any,
) -> tuple[TesterAgent, FakeClient]:
    """Build a tester whose detection and LLM are fully under the test's control."""
    monkeypatch.setattr(shutil, "which", fake_which(*binaries))
    monkeypatch.setattr(tester_module, "_module_available", modules)
    fake = client if client is not None else FakeClient(responses)
    resolved = make_settings(**setting_overrides)
    fake.settings = resolved
    if markers:
        real_exists = Path.exists

        def exists(path: Path) -> bool:
            return path.name in markers or real_exists(path)

        monkeypatch.setattr(Path, "exists", exists)
    return TesterAgent(client=fake, settings=resolved), fake


def run_agent(agent: TesterAgent, goal: str = "make it work", tasks: list[TaskModel] | None = None) -> TestResult:
    """Drive ``agent.run`` without needing an async test plugin."""
    return asyncio.run(agent.run(goal, tasks or []))


def wait_until_reaped(pid: int, *, timeout: float = 2.0) -> bool:
    """True once the pid is gone from /proc, i.e. reaped rather than left a zombie."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        proc_dir = Path(f"/proc/{pid}")
        if not proc_dir.exists():
            return True
        try:
            if (proc_dir / "stat").read_text().split(")")[-1].split()[0] == "Z":
                return False
        except OSError:
            return True
        time.sleep(0.02)
    return False


# --------------------------------------------------------------------------- #
# detect_test_commands
# --------------------------------------------------------------------------- #


def test_only_pytest_is_selected_when_it_is_the_only_probe_hit():
    commands = detect_test_commands(
        which=fake_which("pytest"),
        python="/py",
        module_available=always_false,
        exists=marker_exists("pyproject.toml"),
    )
    assert commands == [["pytest", "-q"]]


def test_other_ecosystems_are_excluded_when_their_binaries_are_absent():
    """Only pytest is installed; npm/pnpm/yarn/cargo/go must not be fabricated."""
    commands = detect_test_commands(
        which=fake_which("pytest"),
        python="/py",
        module_available=always_false,
        exists=marker_exists("pyproject.toml"),
    )
    assert commands == [["pytest", "-q"]]
    for binary in ("npm", "pnpm", "yarn", "cargo", "go"):
        assert not any(binary in argv[0] for argv in commands), commands


def test_installed_binaries_cannot_hijack_a_python_project():
    """Everything is on PATH but only pyproject.toml exists: still just pytest."""
    commands = detect_test_commands(
        which=fake_which("pytest", "npm", "pnpm", "yarn", "cargo", "go", "ruff", "mypy"),
        python="/py",
        module_available=always_true,
        exists=marker_exists("pyproject.toml"),
    )
    assert commands == [["pytest", "-q"]]


@pytest.mark.parametrize(
    ("binaries", "markers", "expected"),
    [
        (("pytest",), ("pytest.ini",), [["pytest", "-q"]]),
        (("pytest",), ("setup.py",), [["pytest", "-q"]]),
        (("pnpm",), ("package.json", "pnpm-lock.yaml"), [["pnpm", "test", "--", "--silent"]]),
        (("yarn",), ("package.json", "yarn.lock"), [["yarn", "test", "--silent"]]),
        (("npm",), ("package.json",), [["npm", "test", "--", "--silent"]]),
        (("cargo",), ("Cargo.toml",), [["cargo", "test", "-q"]]),
        (("go",), ("go.mod",), [["go", "test", "./..."]]),
    ],
)
def test_each_family_needs_its_binary_and_its_marker(binaries, markers, expected):
    commands = detect_test_commands(
        which=fake_which(*binaries),
        python="/py",
        module_available=always_false,
        exists=marker_exists(*markers),
    )
    assert commands == expected


def test_lockfile_runner_and_plain_npm_both_run_for_a_node_project():
    commands = detect_test_commands(
        which=fake_which("npm", "yarn"),
        python="/py",
        module_available=always_false,
        exists=marker_exists("package.json", "yarn.lock"),
    )
    assert commands == [["yarn", "test", "--silent"], ["npm", "test", "--", "--silent"]]


def test_binary_absent_makes_the_whole_family_invisible():
    """package.json/Cargo.toml/go.mod exist but no binary does: not a node/rust/go project."""
    commands = detect_test_commands(
        which=fake_which(),
        python="/py",
        module_available=always_true,
        exists=marker_exists("package.json", "Cargo.toml", "go.mod"),
    )
    assert commands == [["/py", "-m", "pytest", "-q"]]


def test_missing_pytest_module_does_not_fabricate_a_failing_command():
    """The old code ran `python -m pytest` blindly and got "No module named pytest", exit 1."""
    commands = detect_test_commands(
        which=fake_which(),
        python="/py",
        module_available=always_false,
        exists=marker_exists("pyproject.toml", "package.json", "Cargo.toml", "go.mod"),
    )
    assert commands == []


def test_module_fallback_uses_sys_executable():
    commands = detect_test_commands(
        which=fake_which(),
        python="/venv/bin/python3.12",
        module_available=lambda name: name == "pytest",
        exists=marker_exists("pyproject.toml"),
    )
    assert commands == [["/venv/bin/python3.12", "-m", "pytest", "-q"]]


def test_tests_directory_alone_is_enough_of_a_marker():
    commands = detect_test_commands(
        which=fake_which("pytest"),
        python="/py",
        module_available=always_false,
        exists=lambda path: path.name == "test",
    )
    assert commands == [["pytest", "-q"]]


def test_exotic_layout_falls_back_to_installed_tooling():
    """No marker file at all: verify with whatever exists instead of skipping."""
    commands = detect_test_commands(
        which=fake_which("pytest", "go"),
        python="/py",
        module_available=always_false,
        exists=always_false,
    )
    assert commands == [["pytest", "-q"], ["go", "test", "./..."]]


def test_exotic_layout_falls_back_to_the_python_module_last():
    commands = detect_test_commands(
        which=fake_which(),
        python="/py",
        module_available=always_true,
        exists=always_false,
    )
    assert commands == [["/py", "-m", "pytest", "-q"]]


def test_nothing_available_yields_no_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(shutil, "which", fake_which())
    monkeypatch.setattr(tester_module, "_module_available", always_false)
    assert detect_test_commands(workspace=tmp_path, python="/py") == []


def test_detection_defaults_to_shutil_which_and_the_current_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """With no probe injected the module must use shutil.which, not a stub."""
    probed: list[str] = []
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    monkeypatch.setattr(shutil, "which", lambda name: probed.append(name) or None)
    monkeypatch.setattr(tester_module, "_module_available", always_false)

    assert detect_test_commands() == []
    assert "pytest" in probed
    assert "npm" in probed and "cargo" in probed and "go" in probed


def test_detection_does_not_execute_anything_to_find_out(tmp_path: Path):
    """Regression guard: probing must not spawn candidates to see what happens."""
    assert (
        detect_test_commands(
            which=fake_which("pytest", "npm"),
            python="/py",
            module_available=always_false,
            exists=marker_exists("pyproject.toml"),
            workspace=tmp_path,
        )
        == [["pytest", "-q"]]
    )


# --------------------------------------------------------------------------- #
# detect_static_checks
# --------------------------------------------------------------------------- #


def test_static_checks_use_installed_binaries():
    assert detect_static_checks(
        which=fake_which("ruff", "mypy"), python="/py", module_available=always_false
    ) == [["ruff", "check", "."], ["mypy", "."]]


def test_static_checks_are_silently_skipped_when_absent():
    """A missing linter must never become a permanent "fix failures" goal."""
    assert detect_static_checks(which=fake_which(), python="/py", module_available=always_false) == []


def test_static_checks_fall_back_to_the_python_module():
    checks = detect_static_checks(
        which=fake_which(), python="/py", module_available=lambda name: name == "mypy"
    )
    assert checks == [["/py", "-m", "mypy", "."]]


# --------------------------------------------------------------------------- #
# CommandOutcome / VerificationReport
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("fields", "failed", "status"),
    [
        ({"exit_code": 0}, False, "exit 0"),
        ({"exit_code": 1}, True, "exit 1"),
        ({"exit_code": 2, "stderr": "boom"}, True, "exit 2"),
        ({"spawn_error": "FileNotFoundError: nope"}, True, "SPAWN ERROR"),
        ({"timed_out": True, "kill_method": "killpg"}, True, "TIMEOUT"),
        ({"skip_reason": "test phase time budget exhausted"}, False, "SKIPPED"),
        ({"exit_code": NO_TESTS_COLLECTED_EXIT}, False, "vacuous"),
    ],
)
def test_outcome_status_and_failed_property(fields, failed, status):
    result = outcome_for(["pytest", "-q"], **fields)
    assert result.failed is failed
    assert status in result.status


def test_exit_five_is_vacuous_only_for_pytest():
    pytest_outcome = outcome_for(["pytest", "-q"], exit_code=NO_TESTS_COLLECTED_EXIT)
    assert pytest_outcome.failed is False
    assert "no tests collected" in pytest_outcome.status
    # Exit 5 from another tool is still a failure.
    assert outcome_for(["go", "test", "./..."], exit_code=NO_TESTS_COLLECTED_EXIT).failed is True


def test_timeout_status_reports_budget_and_kill_method():
    result = outcome_for(["pytest", "-q"], timed_out=True, timeout=30.0, kill_method="killpg")
    assert result.status == "TIMEOUT after 30s, killed via killpg"
    assert result.failure_line() == "pytest: TIMEOUT after 30s"
    assert "TIMEOUT after 30s" in result.render()


def test_failure_line_carries_label_exit_code_and_the_last_useful_line():
    result = outcome_for(
        ["pytest", "-q"],
        exit_code=1,
        stdout="collected 3 items\n\n",
        stderr="tests/test_a.py:9: AssertionError\n",
    )
    assert result.failure_line() == "pytest: exit 1 -- tests/test_a.py:9: AssertionError"


def test_failure_line_without_output_is_still_identifiable():
    result = outcome_for(["pytest", "-q"], exit_code=3, stdout="   \n")
    assert result.failure_line() == "pytest: exit 3"


def test_spawn_error_failure_line_names_the_error():
    line = outcome_for(["pytest", "-q"], spawn_error="PermissionError: nope").failure_line()
    assert "pytest" in line
    assert "could not be executed" in line and "PermissionError" in line


def test_render_is_a_readable_evidence_row():
    rendered = outcome_for(["pytest", "-q"], exit_code=1, stdout="out line", stderr="err line").render()
    assert rendered.splitlines()[0] == "- `pytest -q` -> exit 1"
    assert "  stdout:" in rendered and "    out line" in rendered
    assert "  stderr:" in rendered and "    err line" in rendered


def test_render_marks_skipped_commands_as_producing_no_evidence():
    rendered = outcome_for(["pytest", "-q"], skip_reason="static-check budget exhausted").render()
    assert "SKIPPED" in rendered
    assert "no evidence" in rendered


def test_report_failure_lines_collects_only_real_failures():
    report = VerificationReport(
        commands=[outcome_for(["pytest", "-q"], exit_code=0)],
        static_checks=[outcome_for(["ruff", "check", "."], exit_code=1, stderr="E501 line too long")],
    )
    assert report.failure_lines() == ["ruff: exit 1 -- E501 line too long"]
    assert report.executed == report.commands + report.static_checks


def test_report_digest_summarizes_every_executed_command():
    report = VerificationReport(
        commands=[outcome_for(["pytest", "-q"], exit_code=0)],
        static_checks=[outcome_for(["mypy", "."], exit_code=2)],
    )
    digest = report.digest()
    assert "pytest: exit 0" in digest
    assert "mypy: exit 2" in digest


def test_report_digest_for_no_runner_is_the_vacuous_summary():
    assert VerificationReport(no_runner=True).digest() == NO_RUNNER_SUMMARY


def test_report_digest_excludes_skipped_commands_but_says_why():
    report = VerificationReport(
        commands=[outcome_for(["pytest", "-q"], skip_reason="test phase time budget exhausted")]
    )
    digest = report.digest()
    assert "SKIPPED" not in digest
    assert "time budget exhausted" in digest


def test_report_render_names_both_sections_including_when_empty():
    rendered = VerificationReport().render()
    assert "Test commands:" in rendered
    assert "- (none detected)" in rendered
    assert "Lint / type checks:" in rendered
    assert "- (no linter installed)" in rendered


# --------------------------------------------------------------------------- #
# Truncation
# --------------------------------------------------------------------------- #


def test_truncation_keeps_the_marker_and_bounds_the_size():
    result = tester_module._truncate("x" * 500_000)
    assert TRUNCATION_MARKER in result
    assert len(result) < MAX_OUTPUT_CHARS + 100
    assert "chars omitted" in result


def test_truncation_honours_an_explicit_limit():
    assert tester_module._truncate("abcdefghij", 4) == f"abcd\n{TRUNCATION_MARKER} (6 chars omitted)"


def test_short_output_is_not_touched():
    assert tester_module._truncate("all good") == "all good"
    assert outcome_for(["pytest", "-q"], stdout="all good").stdout == "all good"


# --------------------------------------------------------------------------- #
# LLM reply parsing
# --------------------------------------------------------------------------- #


def test_extract_json_object_reads_a_fenced_reply():
    assert extract_json_object('```json\n{"passed": true}\n```') == {"passed": True}


def test_extract_json_object_reads_a_bare_or_wrapped_object():
    assert extract_json_object('{"passed": false}') == {"passed": False}
    assert extract_json_object('Verdict:\n{"passed": true}\nthanks!') == {"passed": True}


def test_extract_json_object_returns_none_without_an_object():
    assert extract_json_object("I could not determine the verdict.") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"passed": true}', True),
        ('{"passed": false}', False),
        ('"passed": TRUE', True),
        ("12 passed", True),
        ("3 passed, 1 failed", False),
        ("0 passed, 2 failed", False),
        ("all 42 tests passed", True),
        ("the suite did not pass", False),
        ("tests are unpassed", False),
        ("this was never passed", False),
        ("I could not tell whether anything ran", None),
        ("the output was truncated", None),
    ],
)
def test_heuristic_passed_prefers_the_strongest_signal(text, expected):
    assert heuristic_passed(text) is expected


def test_heuristic_passed_uses_the_final_explicit_marker():
    assert heuristic_passed('draft: {"passed": true}\nfinal: "passed": false') is False


def test_parse_llm_result_returns_a_test_result():
    parsed = parse_llm_result(
        '```json\n{"passed": false, "summary": "1 broke", "failures": ["tests/test_a.py::test_x"]}\n```'
    )
    assert isinstance(parsed, TestResult)
    assert parsed.passed is False
    assert parsed.summary == "1 broke"
    assert parsed.failures == ["tests/test_a.py::test_x"]


def test_parse_llm_result_requires_a_boolean_passed():
    assert parse_llm_result('{"summary": "no verdict field"}') is None
    assert parse_llm_result('{"passed": "true"}') is None


def test_parse_llm_result_returns_none_for_unusable_replies():
    assert parse_llm_result("totally unparseable prose") is None
    assert parse_llm_result("[1, 2, 3]") is None


def test_parse_llm_result_bounds_and_defaults_hostile_payloads():
    parsed = parse_llm_result(json.dumps({"passed": True, "summary": "s" * 5000, "failures": "nope"}))
    assert parsed is not None
    assert len(parsed.summary) < 700  # 600 chars plus the truncation marker
    assert parsed.failures == []
    missing = parse_llm_result('{"passed": true}')
    assert missing is not None
    assert missing.summary == NO_PARSE_SUMMARY


# --------------------------------------------------------------------------- #
# build_env
# --------------------------------------------------------------------------- #


def test_build_env_keeps_only_the_allowlist():
    env = build_env(
        {
            "PATH": "/usr/bin",
            "HOME": "/root",
            "TMPDIR": "/tmp",
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "OPENAI_API_KEY": "sk-openai-secret",
            "FURROW_ANTHROPIC_API_KEY": "sk-furrow-secret",
            "AWS_SECRET_ACCESS_KEY": "aws",
            "GITHUB_TOKEN": "ghp_secret",
        }
    )
    assert env == {"PATH": "/usr/bin", "HOME": "/root", "TMPDIR": "/tmp"}


def test_build_env_reads_the_real_environment_without_leaking_keys():
    env = build_env()
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert not [key for key in env if "SECRET" in key or "TOKEN" in key or key.endswith("_KEY")]


# --------------------------------------------------------------------------- #
# run_command: exit codes, spawn errors, timeouts, cancellation
# --------------------------------------------------------------------------- #


def test_run_command_records_argv_exit_code_and_output(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, Any]] = []

    async def fake_exec(*argv: str, **kwargs: Any) -> FakeProcess:
        calls.append({"argv": argv, **kwargs})
        return FakeProcess(stdout=b"ok\n", stderr=b"warn\n", returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    result = asyncio.run(run_command(["pytest", "-q"], cwd="/tmp", env={"PATH": "/usr/bin"}, timeout=5))
    assert result.argv == ["pytest", "-q"]
    assert result.label == "pytest"
    assert result.exit_code == 0
    assert result.stdout == "ok\n"
    assert result.stderr == "warn\n"
    assert result.failed is False
    assert calls[0]["cwd"] == "/tmp"
    assert calls[0]["env"] == {"PATH": "/usr/bin"}
    assert calls[0]["stdout"] is not None and calls[0]["stderr"] is not None


def test_run_command_truncates_huge_output(monkeypatch: pytest.MonkeyPatch):
    async def fake_exec(*argv: str, **kwargs: Any) -> FakeProcess:
        return FakeProcess(stdout=b"o" * 50_000, returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    result = asyncio.run(run_command(["pytest", "-q"], timeout=5))
    assert TRUNCATION_MARKER in result.stdout
    assert len(result.stdout) < MAX_OUTPUT_CHARS + 100


def test_run_command_reports_a_spawn_error_as_failure_evidence():
    result = asyncio.run(run_command(["furrow-definitely-not-a-real-binary", "-q"], timeout=5))
    assert result.spawn_error is not None
    assert "FileNotFoundError" in result.spawn_error
    assert result.exit_code is None
    assert result.failed is True
    assert "could not be executed" in result.failure_line()


@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX only")
def test_run_command_timeout_kills_the_group_and_leaves_no_zombie(monkeypatch: pytest.MonkeyPatch):
    real_exec = asyncio.create_subprocess_exec
    seen: list[Any] = []

    async def spy(*argv: str, **kwargs: Any) -> Any:
        proc = await real_exec(*argv, **kwargs)
        seen.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    started = time.monotonic()
    result = asyncio.run(run_command([sys.executable, "-c", "import time; time.sleep(60)"], timeout=0.25))
    elapsed = time.monotonic() - started

    assert elapsed < 10, "the timeout must not wait for the child to finish"
    assert result.timed_out is True
    assert result.kill_method == "killpg"
    assert result.failed is True
    assert "TIMEOUT" in result.status
    assert wait_until_reaped(seen[0].pid), "the child was left behind"


def test_kill_falls_back_to_proc_kill_when_killpg_is_impossible(monkeypatch: pytest.MonkeyPatch):
    """A pid that no longer exists makes os.getpgid raise, so proc.kill() must run."""
    proc = FakeProcess(hang=True, returncode=-9)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *a, **k: spawned(proc))
    result = asyncio.run(run_command(["pytest", "-q"], timeout=0.05))
    assert result.timed_out is True
    assert result.kill_method == "kill"
    assert proc.kill_calls == 1
    assert result.exit_code == -9
    assert result.failed is True


def test_reap_lets_cancellation_through():
    """A shutdown arriving while we reap must unwind, not be swallowed."""

    async def scenario() -> None:
        proc = FakeProcess(wait_hangs=True)
        task = asyncio.create_task(tester_module._reap(proc))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_run_command_reraises_cancellation_and_kills_the_child(monkeypatch: pytest.MonkeyPatch):
    proc = FakeProcess(hang=True, returncode=-9)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *a, **k: spawned(proc))

    async def scenario() -> None:
        task = asyncio.create_task(run_command(["pytest", "-q"], timeout=30))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert proc.kill_calls == 1, "a cancelled run must still kill the child"


# --------------------------------------------------------------------------- #
# TesterAgent: evidence -> verdict
# --------------------------------------------------------------------------- #


def test_no_runner_is_a_vacuous_pass_not_a_failure(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    agent, client = make_agent(
        monkeypatch, binaries=(), markers=(), modules=always_false, responses=[verdict_json(False, "fake")]
    )
    result = run_agent(agent)

    assert result.passed is True
    assert result.failures == []
    assert result.summary == NO_RUNNER_SUMMARY
    assert agent.last_kind == NO_TEST_RUNNER_KIND
    assert agent.no_test_runner is True
    assert runner.calls == [], "nothing may be executed when there is no runner"
    assert client.prompts == [], "no verdict is needed when there is nothing to verify"


def test_no_runner_summary_carries_no_fix_request_vocabulary():
    assert not [word for word in FAILURE_WORDS if word in NO_RUNNER_SUMMARY.lower()]
    assert "no-test-runner" in NO_RUNNER_SUMMARY


def test_no_runner_state_is_recoverable_from_the_agent_not_from_prose(
    monkeypatch: pytest.MonkeyPatch, runner: Recorder
):
    """Before the first run there is no state; after a vacuous run the flag flips."""
    agent, _ = make_agent(monkeypatch, binaries=(), markers=(), modules=always_false)
    assert agent.last_kind is None
    assert agent.no_test_runner is False
    run_agent(agent)
    assert agent.no_test_runner is True


def test_non_zero_exit_is_failure_evidence_even_when_the_llm_says_pass(
    monkeypatch: pytest.MonkeyPatch, runner: Recorder
):
    runner.outcomes["pytest"] = {
        "exit_code": 1,
        "stdout": "collected 2 items\n",
        "stderr": "tests/test_a.py:9: in test_x\n    assert 0\nE   AssertionError\n1 failed, 1 passed\n",
    }
    agent, client = make_agent(monkeypatch, responses=[verdict_json(True, "looks green to me")])
    result = run_agent(agent)

    assert result.passed is False
    assert agent.last_kind == TEST_RESULT_KIND_RAN
    assert len(result.failures) == 1
    assert result.failures[0] == "pytest: exit 1 -- 1 failed, 1 passed"
    assert "[overridden-by-exit-codes]" in result.summary
    assert "looks green to me" in result.summary
    assert "E   AssertionError" in client.prompts[0]


def test_failure_evidence_reaches_the_prompt(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    runner.outcomes["pytest"] = {"exit_code": 1, "stderr": "E   AssertionError: nope\n"}
    agent, client = make_agent(monkeypatch, responses=[verdict_json(False, "broken")])
    result = run_agent(agent)

    prompt = client.prompts[0]
    assert "`pytest -q` -> exit 1" in prompt
    assert "    E   AssertionError: nope" in prompt
    assert "exit code 5" in prompt  # the vacuous-exit rule is spelled out for the model
    assert "pytest: exit 1" in result.summary


def test_exit_zero_everywhere_passes_when_the_llm_passes(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    agent, _ = make_agent(monkeypatch, responses=[verdict_json(True, "all green")])
    result = run_agent(agent)
    assert result.passed is True
    assert result.failures == []
    assert "all green" in result.summary
    assert "[evidence] pytest: exit 0" in result.summary


def test_exit_zero_everywhere_passes_when_the_llm_is_unparseable(
    monkeypatch: pytest.MonkeyPatch, runner: Recorder
):
    agent, _ = make_agent(monkeypatch, responses=["I am not sure, sorry."])
    result = run_agent(agent)
    assert result.passed is True
    assert result.failures == []
    assert NO_PARSE_SUMMARY in result.summary


def test_llm_failure_list_survives_a_clean_exit(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    """Exit codes cannot explain a logic failure; the LLM list must still be handed on."""
    agent, _ = make_agent(
        monkeypatch,
        responses=[verdict_json(False, "assertion is wrong", ["tests/test_a.py::test_x AssertionError"])],
    )
    result = run_agent(agent)
    assert result.passed is False
    assert result.failures == ["tests/test_a.py::test_x AssertionError"]
    assert "[overridden-by-exit-codes]" not in result.summary


def test_exit_code_five_is_vacuous_and_does_not_fail_the_cycle(
    monkeypatch: pytest.MonkeyPatch, runner: Recorder
):
    runner.outcomes["pytest"] = {"exit_code": NO_TESTS_COLLECTED_EXIT, "stdout": "no tests ran\n"}
    agent, client = make_agent(monkeypatch, responses=[verdict_json(True, "nothing collected")])
    result = run_agent(agent)

    assert result.passed is True
    assert result.failures == []
    assert "vacuous" in client.prompts[0]
    assert "no tests collected" in result.summary


def test_exit_code_five_still_fails_for_a_non_pytest_tool(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    runner.outcomes["go"] = {"exit_code": NO_TESTS_COLLECTED_EXIT, "stdout": "no test files\n"}
    agent, _ = make_agent(
        monkeypatch,
        binaries=("pytest", "go"),
        markers=("pyproject.toml", "go.mod"),
        responses=[verdict_json(True, "ok")],
    )
    result = run_agent(agent)
    assert result.passed is False
    assert [line for line in result.failures if line.startswith("go test: exit 5")]


def test_timeout_is_failure_evidence(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    runner.outcomes["pytest"] = {
        "exit_code": -9,
        "timed_out": True,
        "kill_method": "killpg",
        "timeout": 30.0,
    }
    agent, client = make_agent(monkeypatch, responses=[verdict_json(True, "sure, passed")])
    result = run_agent(agent)

    assert result.passed is False
    assert result.failures == ["pytest: TIMEOUT after 30s"]
    assert "TIMEOUT after 30s" in result.summary
    assert "TIMEOUT after 30s, killed via killpg" in client.prompts[0]


def test_spawn_error_is_failure_evidence(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    runner.outcomes["pytest"] = {"spawn_error": "FileNotFoundError: pytest", "exit_code": None}
    agent, _ = make_agent(monkeypatch, responses=[verdict_json(True, "sure")])
    result = run_agent(agent)
    assert result.passed is False
    assert "could not be executed" in result.failures[0]


def test_static_checks_appear_in_evidence_and_failures(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    runner.outcomes["ruff"] = {"exit_code": 1, "stderr": "furrow/x.py:1:1: E501 line too long\n"}
    runner.outcomes["mypy"] = {"exit_code": 0}
    agent, client = make_agent(
        monkeypatch, binaries=("pytest", "ruff", "mypy"), responses=[verdict_json(True, "ok")]
    )
    result = run_agent(agent)

    assert runner.argvs == [["pytest", "-q"], ["ruff", "check", "."], ["mypy", "."]]
    assert result.passed is False
    assert result.failures == ["ruff: exit 1 -- furrow/x.py:1:1: E501 line too long"]
    prompt = client.prompts[0]
    assert "`ruff check .` -> exit 1" in prompt
    assert "`mypy .` -> exit 0" in prompt
    assert "ruff: exit 1" in result.summary


def test_absent_static_checks_are_skipped_silently(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    agent, client = make_agent(monkeypatch, binaries=("pytest",), responses=[verdict_json(True, "ok")])
    result = run_agent(agent)

    assert runner.argvs == [["pytest", "-q"]]
    assert result.passed is True
    assert not [line for line in result.failures if "ruff" in line or "mypy" in line]
    assert "- (no linter installed)" in client.prompts[0]


def test_budget_exhaustion_skips_the_remaining_commands(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    clock = FakeClock()
    monkeypatch.setattr(tester_module, "time", clock)
    runner.on_call = lambda _argv, _kwargs: clock.advance(DEFAULT_TEST_TIMEOUT * TEST_PHASE_BUDGET_FACTOR + 1)
    agent, client = make_agent(
        monkeypatch,
        binaries=("pytest", "go"),
        markers=("pyproject.toml", "go.mod"),
        test_timeout=5,
        responses=[verdict_json(True, "ok")],
    )
    result = run_agent(agent)

    assert runner.argvs == [["pytest", "-q"]], "the second command is skipped, never run"
    prompt = client.prompts[0]
    assert "`go test ./...` -> SKIPPED (test phase time budget exhausted)" in prompt
    assert "no evidence" in prompt
    assert result.passed is True  # nothing failed; the second command produced no evidence
    assert "SKIPPED" not in result.summary


def test_commands_get_the_workspace_env_and_per_command_timeout(
    monkeypatch: pytest.MonkeyPatch, runner: Recorder, tmp_path: Path
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    agent, _ = make_agent(monkeypatch, test_timeout=7, workspace=tmp_path)
    run_agent(agent)

    call = runner.calls[0]
    assert call["cwd"] == tmp_path
    assert 0 < call["timeout"] <= 7
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert call["env"].get("PATH") == os.environ.get("PATH")


def test_test_timeout_settings_are_sanitized():
    assert TesterAgent(client=FakeClient(), settings=make_settings(test_timeout=30)).test_timeout == 30.0
    assert TesterAgent(client=FakeClient(), settings=make_settings(test_timeout=0)).test_timeout == 120.0

    class BareSettings:
        """Settings without ``test_timeout`` (older Settings, or a stub)."""

    assert TesterAgent(client=FakeClient(), settings=BareSettings()).test_timeout == DEFAULT_TEST_TIMEOUT
    assert DEFAULT_TEST_TIMEOUT == 120

    class WeirdSettings:
        test_timeout = "soon"

    assert TesterAgent(client=FakeClient(), settings=WeirdSettings()).test_timeout == 120.0


def test_settings_are_taken_from_the_client_when_not_passed():
    settings = make_settings(test_timeout=9)
    agent = TesterAgent(client=FakeClient(settings=settings))
    assert agent.settings is settings
    assert agent.test_timeout == 9.0
    assert TesterAgent(settings=make_settings()).client is not None


def test_prompt_contains_goal_task_results_and_the_evidence_table(
    monkeypatch: pytest.MonkeyPatch, runner: Recorder
):
    agent, client = make_agent(monkeypatch)
    tasks = [
        TaskModel(id="t1", description="add the route", status="done", result="created app.py"),
        TaskModel(id="t2", description="fix the parser", status="failed", result="E   SyntaxError"),
    ]
    run_agent(agent, goal="expose GET /health", tasks=tasks)

    prompt = client.prompts[0]
    assert prompt.startswith(TESTER_PROMPT)  # the rewritten prompts.py still resolves
    assert "Goal: expose GET /health" in prompt
    assert "[t1] done: add the route -> created app.py" in prompt
    assert "[t2] failed: fix the parser -> E   SyntaxError" in prompt
    assert "Test commands:" in prompt and "Lint / type checks:" in prompt
    assert "`pytest -q` -> exit 0" in prompt
    assert client.models == ["claude-3-5-sonnet-20241022"]


def test_prompt_without_task_results_says_so(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    agent, client = make_agent(monkeypatch)
    run_agent(agent, tasks=[])
    assert "- (no task results available)" in client.prompts[0]


def test_prompt_bounds_goal_and_task_result_size(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    agent, client = make_agent(monkeypatch)
    tasks = [TaskModel(id=f"t{i}", description="d", result="R" * 5000) for i in range(25)]
    run_agent(agent, goal="G" * 9000, tasks=tasks)

    prompt = client.prompts[0]
    assert "G" * 9000 not in prompt
    assert "R" * 5000 not in prompt
    assert TRUNCATION_MARKER in prompt
    assert f"[t{MAX_PROMPT_TASKS - 1}]" in prompt
    assert f"[t{MAX_PROMPT_TASKS}]" not in prompt
    assert "5 more task(s) omitted" in prompt


def test_llm_outage_does_not_fabricate_a_test_failure(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    client = SpyClient(RuntimeError("provider is down"))
    agent, _ = make_agent(monkeypatch, client=client, responses=[verdict_json(True, "unused")])
    result = run_agent(agent)

    assert result.passed is True
    assert "[llm-unavailable] RuntimeError: provider is down" in result.summary
    assert agent.last_kind == TEST_RESULT_KIND_ERROR


def test_llm_outage_cannot_hide_an_objective_failure(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    runner.outcomes["pytest"] = {"exit_code": 1, "stderr": "E   AssertionError\n"}
    agent, _ = make_agent(monkeypatch, client=SpyClient(RuntimeError("provider is down")))
    result = run_agent(agent)

    assert result.passed is False
    assert result.failures[0].startswith("pytest: exit 1")
    assert agent.last_kind == TEST_RESULT_KIND_ERROR


def test_llm_receives_the_tester_model(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    agent, client = make_agent(monkeypatch, tester_model="stub-tester-model")
    run_agent(agent)
    assert client.models == ["stub-tester-model"]


def test_unexpected_runner_exception_becomes_a_failed_result(monkeypatch: pytest.MonkeyPatch):
    async def boom(self: TesterAgent):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(TesterAgent, "_run_tests", boom)
    agent, client = make_agent(monkeypatch, responses=[verdict_json(True, "ok")])
    result = run_agent(agent)

    assert result.passed is False
    assert "RuntimeError: disk on fire" in result.summary
    assert result.failures == [result.summary]
    assert agent.last_kind == TEST_RESULT_KIND_ERROR
    assert client.prompts == []


def test_cancellation_propagates_instead_of_becoming_a_failure(monkeypatch: pytest.MonkeyPatch):
    async def cancelled(self: TesterAgent):
        raise asyncio.CancelledError

    monkeypatch.setattr(TesterAgent, "_run_tests", cancelled)
    agent, _ = make_agent(monkeypatch)
    with pytest.raises(asyncio.CancelledError):
        run_agent(agent)
    assert agent.last_kind is None


def test_cancellation_during_the_llm_call_propagates(monkeypatch: pytest.MonkeyPatch, runner: Recorder):
    async def cancelled(prompt: str, system: str = "", model: str | None = None) -> str:
        raise asyncio.CancelledError

    client = FakeClient()
    client.complete = cancelled  # type: ignore[method-assign]
    agent, _ = make_agent(monkeypatch, client=client)
    with pytest.raises(asyncio.CancelledError):
        run_agent(agent)
    assert agent.last_kind is None


def test_evidence_wins_when_the_verdict_object_has_the_wrong_shape(
    monkeypatch: pytest.MonkeyPatch, runner: Recorder
):
    runner.outcomes["pytest"] = {"exit_code": 2, "stderr": "collection error\n"}
    agent, _ = make_agent(monkeypatch, responses=['{"verdict": "green", "notes": "nothing red"}'])
    result = run_agent(agent)
    assert result.passed is False
    assert result.failures[0].startswith("pytest: exit 2")