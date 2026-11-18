from __future__ import annotations

import asyncio

from furrow.agents.tester import TesterAgent, _looks_like_failure, resolve_test_timeout
from furrow.config import Settings, TestResult


class FakeLLMClient:
    def __init__(self, response="", raises=None):
        self.response = response
        self.raises = raises
        self.prompts = []
        self.models = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        self.models.append(model)
        if self.raises:
            raise self.raises
        return self.response


def fake_proc(returncode=0, stdout=b"", stderr=b""):
    """Build a stand-in for an asyncio subprocess with a fixed exit status."""

    class FakeProc:
        pid = 4242

        async def communicate(self):
            return stdout, stderr

        async def wait(self):
            return returncode

        def kill(self):
            return None

    FakeProc.returncode = returncode
    return FakeProc()


def run(coro):
    """Run a coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


def test_detect_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    cmd = run(agent.detect_command())
    assert cmd is not None and "pytest" in cmd


def test_detect_packagejson(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    cmd = run(agent.detect_command())
    assert cmd is not None and "npm" in cmd


def test_detect_command_unreadable_package_json_falls_back(tmp_path):
    """An unreadable / non-UTF-8 package.json must not crash detection.

    Invalid UTF-8 bytes should be logged and the npm/pnpm/yarn fallback
    (based on lockfiles) should still be returned.
    """
    (tmp_path / "package.json").write_bytes(b"\xff\xfe\x00")
    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    cmd = run(agent.detect_command())
    assert cmd is not None
    assert cmd == ["npm", "test", "--silent"]


def test_detect_command_valid_package_json_with_test_script(tmp_path):
    """A valid package.json with a test script returns the npm fallback."""
    (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest"}}')
    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    cmd = run(agent.detect_command())
    assert cmd is not None
    assert cmd == ["npm", "test", "--silent"]


def test_detect_cargo(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    cmd = run(agent.detect_command())
    assert cmd is not None and "cargo" in cmd


def test_detect_empty(tmp_path):
    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    cmd = run(agent.detect_command())
    assert cmd is None


def test_run_no_runner_is_reported_as_skipped(tmp_path):
    """No detected test runner means a skipped run, not a passed one."""
    agent = TesterAgent(
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )
    res = run(agent.run("goal", []))
    assert isinstance(res, TestResult)
    assert res.passed is True
    assert res.skipped is True
    assert "skipped" in res.summary.lower()
    assert "no test runner" in res.summary.lower()
    assert res.failures == []


def test_run_llm_result(tmp_path):
    client = FakeLLMClient(response='{"passed": false, "summary": "boom", "failures": ["x"]}')
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )
    # create pyproject so detect_command returns pytest; empty workspace -> None.
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    # pytest with no tests collects 0 -> exit 0.
    res = run(agent.run("goal", []))
    assert res.passed is False
    assert res.failures == ["x"]


def test_run_llm_raises_fallback(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    client = FakeLLMClient(raises=RuntimeError("boom"))
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )
    res = run(agent.run("goal", []))
    assert isinstance(res, TestResult)


def test_run_failed_command_uses_local_fallback_classification(tmp_path, monkeypatch):
    """A failing test command is reported as failed even when the LLM misbehaves.

    The LLM classification raises, so the local fallback in
    ``furrow.agents.tester`` has to decide the outcome from the raw output.
    """
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    stderr_text = (
        "=================================== FAILURES ===================================\n"
        "FAILED tests/test_x.py::test_y - AssertionError: expected 1 got 2\n"
        "=========================== 1 failed, 2 passed in 0.12s ===========================\n"
    )

    class FakeProc:
        pid = 4242
        returncode = 1

        async def communicate(self):
            return b"", stderr_text.encode()

        async def wait(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(raises=RuntimeError("llm down"))
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))

    assert isinstance(res, TestResult)
    assert res.passed is False
    assert res.skipped is False
    assert res.failures
    assert any("test_y" in failure for failure in res.failures)
    assert "test_y" in res.summary


def test_run_timeout_kills_process_group(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    class FakeProc:
        pid = 12345
        returncode = None

        async def communicate(self):
            await asyncio.sleep(10)
            return b"", b""

        async def wait(self):
            self.returncode = -9
            return self.returncode

        def kill(self):
            self.returncode = -9

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProc()

    killed = {"pg": False, "kill": False}

    def fake_killpg(pgid, sig):
        killed["pg"] = True
        raise OSError("no pg")

    def fake_kill(pid, sig=None):
        killed["kill"] = True

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr("os.killpg", fake_killpg)
    monkeypatch.setattr("os.getpgid", lambda pid: pid)
    monkeypatch.setattr("os.kill", fake_kill)

    # The test-suite timeout comes from `test_timeout`, not from the LLM
    # request timeout, so the setting has to be provided explicitly here.
    settings = Settings(_env_file=None, workspace=tmp_path, request_timeout=30)
    settings = settings.model_copy(update={"test_timeout": 1})
    agent = TesterAgent(settings=settings, workspace=tmp_path)
    res = run(agent.run("goal", []))
    assert isinstance(res, TestResult)
    assert res.passed is False
    assert "timed out" in res.summary.lower()
    assert killed["pg"] is True or killed["kill"] is True


def test_failed_command_is_never_reported_as_passed(tmp_path, monkeypatch):
    """A non-zero exit status wins over a green LLM verdict.

    The exit status of the test command is the ground truth: a command that
    exited non-zero ran failing (or crashing) tests, so the result must not be
    reported as passed even when the LLM claims everything is fine.
    """
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    stderr_text = (
        "=================================== FAILURES ===================================\n"
        "FAILED tests/test_x.py::test_y - AssertionError: expected 1 got 2\n"
        "=========================== 1 failed, 2 passed in 0.12s ===========================\n"
    )

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc(returncode=1, stderr=stderr_text.encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(response='{"passed": true, "summary": "everything is fine"}')
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))

    assert res.passed is False
    assert res.skipped is False
    assert "failed" in res.summary.lower()
    assert res.failures
    assert any("test_y" in failure for failure in res.failures)


def test_green_command_with_green_verdict_is_passed(tmp_path, monkeypatch):
    """A zero exit status plus a green LLM verdict is a pass."""
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc(returncode=0, stdout=b"5 passed in 0.12s\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(response='{"passed": true, "summary": "all green"}')
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))

    assert res.passed is True
    assert res.skipped is False
    assert res.failures == []


def test_green_command_with_failing_verdict_is_not_passed(tmp_path, monkeypatch):
    """A zero exit status does not overrule an LLM that says the suite failed."""
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc(returncode=0, stdout=b"no tests ran\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(response='{"passed": false, "summary": "the suite did not run"}')
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))

    assert res.passed is False
    assert "did not run" in res.summary.lower()


def test_spawn_failure_is_reported_as_an_infrastructure_error(tmp_path, monkeypatch):
    """A runner that cannot be started is not reported as a test failure."""
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    async def fake_create_subprocess_exec(*args, **kwargs):
        raise FileNotFoundError("pytest")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(response='{"passed": true, "summary": "all green"}')
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))

    assert res.passed is False
    assert res.skipped is False
    assert "could not be started" in res.summary.lower()
    assert "did not run" in res.summary.lower()
    assert res.failures
    assert any("could not be started" in failure for failure in res.failures)


def test_test_command_timeout_comes_from_test_timeout(tmp_path, monkeypatch):
    """The test-suite timeout comes from `test_timeout`, not `request_timeout`."""
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    recorded = {}

    async def fake_wait_for(awaitable, timeout=None):
        recorded["timeout"] = timeout
        return await awaitable

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc(returncode=0, stdout=b"5 passed in 0.12s\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    settings = Settings(_env_file=None, workspace=tmp_path, request_timeout=1.0)
    settings = settings.model_copy(update={"test_timeout": 600.0})
    agent = TesterAgent(
        client=FakeLLMClient(response='{"passed": true, "summary": "all green"}'),
        settings=settings,
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))

    assert recorded["timeout"] == 600.0
    assert resolve_test_timeout(settings) == 600.0
    assert res.passed is True


def test_test_timeout_defaults_to_a_dedicated_value(tmp_path):
    """Without an explicit `test_timeout` the tester still gets its own timeout."""
    settings = Settings(_env_file=None, workspace=tmp_path, request_timeout=1.0)
    assert resolve_test_timeout(settings) == 600.0


def test_looks_like_failure_ignores_zero_counters():
    """Negated / zero-count phrases are not failure lines."""
    assert _looks_like_failure("5 passed, 0 failed in 0.12s") is False
    assert _looks_like_failure("no failures") is False
    assert _looks_like_failure("0 errors") is False
    assert _looks_like_failure("failed: 0") is False
    assert _looks_like_failure("FAILED (errors=0)") is False
    assert _looks_like_failure("1 failed, 4 passed in 0.12s") is True
    assert _looks_like_failure("FAILED tests/test_x.py::test_y - AssertionError") is True


def test_looks_like_failure_catches_crash_signals():
    """Strong crash signals must be detected even when the LLM is unavailable."""
    assert _looks_like_failure("thread 'main' panicked at src/main.rs:1:1") is True
    assert _looks_like_failure("Segmentation fault (core dumped)") is True
    assert _looks_like_failure("fatal error: unable to open database") is True
    assert _looks_like_failure("0 errors") is False
    assert _looks_like_failure("5 passed, 0 failed in 0.12s") is False


def test_green_summary_line_is_not_reported_as_a_failure(tmp_path, monkeypatch):
    """'5 passed, 0 failed' must not turn into a failure entry."""
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc(returncode=0, stdout=b"5 passed, 0 failed in 0.12s\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(raises=RuntimeError("llm down"))
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))

    assert res.passed is True
    assert res.failures == []


def test_truncation_keeps_tail_with_summary_marker(tmp_path, monkeypatch):
    """Output longer than 20000 chars keeps the tail (with summary) and adds marker."""
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    head_start = "START_OF_ORIGINAL_OUTPUT_ABCDEFG"
    padding = ("Y" * 10000) + ("Z" * 20000)
    summary = (
        "\n=========================== 2 failed, 3 passed in 1.23s "
        "===========================\n"
    )
    stdout = (head_start + padding + summary).encode()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc(returncode=1, stdout=stdout)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(raises=RuntimeError("llm down"))
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))
    assert res.passed is False
    assert "[... output truncated ...]" in res.summary
    assert "2 failed" in " ".join(res.failures)
    # Beginning of original output is no longer what is kept
    assert head_start not in res.summary
    assert head_start not in " ".join(res.failures)
    assert client.prompts
    prompt = client.prompts[0]
    assert "[... output truncated ...]" in prompt
    assert "2 failed" in prompt
    assert head_start not in prompt


 
def test_tester_run_exception_includes_message_in_failures(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    async def fake_create_subprocess_exec(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    agent = TesterAgent(
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))
    assert isinstance(res, TestResult)
    assert res.passed is False
    assert "RuntimeError" in res.summary
    assert res.failures
    assert any("RuntimeError" in f for f in res.failures)

    """Old head-only behaviour is gone - beginning of output is not kept."""
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    head_start = "UNIQUE_HEAD_MARKER_111"
    filler = "M" * 25005
    tail = (
        "\n=========================== 2 failed, 5 passed in 1.23s "
        "===========================\nUNIQUE_TAIL_MARKER_777\n"
    )
    stdout = (head_start + filler + tail).encode()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc(returncode=1, stdout=stdout)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(raises=RuntimeError("llm down"))
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))
    assert res.passed is False
    assert "[... output truncated ...]" in res.summary
    assert "2 failed" in " ".join(res.failures)
    assert head_start not in res.summary
    assert head_start not in " ".join(res.failures)
    assert client.prompts
    prompt = client.prompts[0]
    assert "[... output truncated ...]" in prompt
    assert "2 failed" in prompt
    assert "UNIQUE_TAIL_MARKER_777" in prompt
    assert head_start not in prompt


def test_green_command_contradictory_verdict_is_failed(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc(returncode=0, stdout=b"5 passed in 0.12s\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(response='{"passed": true, "summary": "all good", "failures": ["boom"]}')
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))

    assert res.passed is False
    assert "boom" in res.failures
    assert "pass" in res.summary.lower()
    assert "fail" in res.summary.lower()


def test_green_command_green_verdict_regression(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake_proc(returncode=0, stdout=b"5 passed in 0.12s\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(response='{"passed": true, "summary": "all green"}')
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    res = run(agent.run("goal", []))

    assert res.passed is True


def test_test_command_does_not_inherit_stdin(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\n")

    captured = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured.update(kwargs)
        return fake_proc(returncode=0, stdout=b"5 passed in 0.12s\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    client = FakeLLMClient(response='{"passed": true, "summary": "all green"}')
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    run(agent.run("goal", []))

    assert captured.get("stdin") is asyncio.subprocess.DEVNULL


# --- Tests for uncovered lines in tester.py ---

def test_resolve_test_timeout_invalid_values(tmp_path):
    """Test resolve_test_timeout with various invalid values."""
    settings = Settings(_env_file=None, workspace=tmp_path, request_timeout=30)

    # Default
    assert resolve_test_timeout(settings) == 600.0

    # Invalid type (None)
    settings = settings.model_copy(update={"test_timeout": None})
    assert resolve_test_timeout(settings) == 600.0

    # Invalid type (string)
    settings = settings.model_copy(update={"test_timeout": "invalid"})
    assert resolve_test_timeout(settings) == 600.0

    # Negative value
    settings = settings.model_copy(update={"test_timeout": -1})
    assert resolve_test_timeout(settings) == 600.0

    # Zero value
    settings = settings.model_copy(update={"test_timeout": 0})
    assert resolve_test_timeout(settings) == 600.0

    # Valid positive value
    settings = settings.model_copy(update={"test_timeout": 120.0})
    assert resolve_test_timeout(settings) == 120.0


def test_detect_command_cargo(tmp_path, monkeypatch):
    """Test detect_command for Cargo projects."""
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'test'\n")
    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    cmd = run(agent.detect_command())
    assert cmd is not None
    assert "cargo" in cmd
    assert "test" in cmd


def test_detect_command_go(tmp_path, monkeypatch):
    """Test detect_command for Go projects."""
    (tmp_path / "go.mod").write_text("module test\n")
    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    cmd = run(agent.detect_command())
    assert cmd is not None
    assert "go" in cmd
    assert "test" in cmd


def test_detect_command_python_markers(tmp_path):
    """Test detect_command falls back to pytest for Python markers."""
    for marker in ["pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini", "requirements.txt"]:
        (tmp_path / marker).write_text("")
        agent = TesterAgent(
            settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path
        )
        cmd = run(agent.detect_command())
        assert cmd is not None
        assert "pytest" in cmd
        (tmp_path / marker).unlink()


def test_parse_test_output_file_line_col(tmp_path):
    """Test parse_test_output with file:line:col pattern."""
    from furrow.agents.tester import parse_test_output

    output = "test_file.py:10:5: error message"
    failures = parse_test_output(output)
    assert len(failures) == 1
    assert failures[0]["file"] == "test_file.py"
    assert failures[0]["line"] == 10
    assert failures[0]["test"] is None
    assert failures[0]["error"] == ""


def test_parse_test_output_file_line(tmp_path):
    """Test parse_test_output with file:line pattern."""
    from furrow.agents.tester import parse_test_output

    output = "test_file.py:10"
    failures = parse_test_output(output)
    assert len(failures) == 1
    assert failures[0]["file"] == "test_file.py"
    assert failures[0]["line"] == 10


def test_parse_test_output_exception_no_failures(tmp_path):
    """Test parse_test_output extracts exceptions when no other failures."""
    from furrow.agents.tester import parse_test_output

    output = "ValueError: invalid value\nTypeError: bad type"
    failures = parse_test_output(output)
    assert len(failures) == 2
    assert failures[0]["error"] == "ValueError: invalid value"
    assert failures[1]["error"] == "TypeError: bad type"


def test_parse_test_output_exception_with_failures(tmp_path):
    """Test parse_test_output adds exception to existing failures."""
    from furrow.agents.tester import parse_test_output

    output = "FAILED test_file.py::test_func - AssertionError: failed\nValueError: invalid value"
    failures = parse_test_output(output)
    assert len(failures) == 1
    assert failures[0]["test"] == "test_func"
    assert failures[0]["error"] == "ValueError: invalid value"


def test_finalize_llm_none_fallback(tmp_path):
    """Test _finalize when llm_result is None."""
    from furrow.agents.tester import TesterAgent

    agent = TesterAgent(
        settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path
    )
    result = run(agent._finalize(
        llm_result=None,
        returncode=1,
        combined="FAILED test_x - AssertionError",
        spawn_failed=False,
        spawn_error="",
        timed_out=False,
        timeout=60.0,
    ))
    assert result.passed is False
    assert "FAILED" in result.summary
    assert result.failures


def test_finalize_spawn_failed(tmp_path):
    """Test _finalize when spawn_failed is True."""
    from furrow.agents.tester import TesterAgent
    from furrow.config import TestResult

    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    result = run(agent._finalize(
        llm_result=TestResult(passed=True, summary="all good", failures=[]),
        returncode=1,
        combined="",
        spawn_failed=True,
        spawn_error="FileNotFoundError: pytest",
        timed_out=False,
        timeout=60.0,
    ))
    assert result.passed is False
    assert "could not be started" in result.summary
    assert "did not run" in result.summary


def test_finalize_timed_out(tmp_path):
    """Test _finalize when timed_out is True."""
    from furrow.agents.tester import TesterAgent
    from furrow.config import TestResult

    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    result = run(agent._finalize(
        llm_result=TestResult(passed=True, summary="all good", failures=[]),
        returncode=1,
        combined="",
        spawn_failed=False,
        spawn_error="",
        timed_out=True,
        timeout=60.0,
    ))
    assert result.passed is False
    assert "timed out" in result.summary.lower()


def test_finalize_llm_pass_returncode_nonzero(tmp_path):
    """Test _finalize when LLM says pass but returncode is non-zero."""
    from furrow.agents.tester import TesterAgent
    from furrow.config import TestResult

    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    result = run(agent._finalize(
        llm_result=TestResult(passed=True, summary="all good", failures=[]),
        returncode=1,
        combined="FAILED test_x",
        spawn_failed=False,
        spawn_error="",
        timed_out=False,
        timeout=60.0,
    ))
    assert result.passed is False
    assert "authoritative" in result.summary


def test_finalize_llm_fail_returncode_zero(tmp_path):
    """Test _finalize when LLM says fail but returncode is zero."""
    from furrow.agents.tester import TesterAgent
    from furrow.config import TestResult

    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    result = run(agent._finalize(
        llm_result=TestResult(passed=False, summary="suite did not run", failures=["x"]),
        returncode=0,
        combined="",
        spawn_failed=False,
        spawn_error="",
        timed_out=False,
        timeout=60.0,
    ))
    assert result.passed is False
    assert "LLM reported a failure" in result.summary


def test_finalize_llm_pass_with_failures_returncode_zero(tmp_path):
    """Test _finalize when LLM says pass but lists failures and returncode is zero."""
    from furrow.agents.tester import TesterAgent
    from furrow.config import TestResult

    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    result = run(agent._finalize(
        llm_result=TestResult(passed=True, summary="all good", failures=["test_x failed"]),
        returncode=0,
        combined="",
        spawn_failed=False,
        spawn_error="",
        timed_out=False,
        timeout=60.0,
    ))
    assert result.passed is False
    assert "listed failing tests" in result.summary


def test_extract_failures_complex(tmp_path):
    """Test _extract_failures with complex output."""
    from furrow.agents.tester import _extract_failures

    output = """
    FAILED tests/test_a.py::test_one - AssertionError: expected 1
    FAILED tests/test_b.py::test_two - ValueError: invalid
    tests/test_c.py:42:56
    tests/test_d.py:100
    RuntimeError: something broke
    5 passed, 2 failed in 0.12s
    """
    failures = _extract_failures(output)
    assert len(failures) >= 3
    assert any("test_one" in f for f in failures)
    assert any("test_two" in f for f in failures)
    assert any("RuntimeError" in f for f in failures)


def test_has_failure_keywords(tmp_path):
    """Test _has_failure_keywords."""
    from furrow.agents.tester import _has_failure_keywords

    assert _has_failure_keywords("FAILED test_x") is True
    assert _has_failure_keywords("Error: something") is True
    assert _has_failure_keywords("5 passed, 0 failed") is False
    assert _has_failure_keywords("no failures") is False
    assert _has_failure_keywords("") is False


def test_looks_like_failure_edge_cases(tmp_path):
    """Test _looks_like_failure edge cases."""
    from furrow.agents.tester import _looks_like_failure

    # Various failure keywords
    assert _looks_like_failure("panic: something") is True
    assert _looks_like_failure("segfault") is True
    assert _looks_like_failure("segmentation fault") is True
    assert _looks_like_failure("core dumped") is True
    assert _looks_like_failure("fatal error") is True
    assert _looks_like_failure("assertion") is True
    assert _looks_like_failure("traceback") is True

    # Zero counters
    assert _looks_like_failure("0 failed") is False
    assert _looks_like_failure("failed: 0") is False
    assert _looks_like_failure("FAILED (errors=0)") is False
    assert _looks_like_failure("5 passed, 0 failed") is False

    # Real failures
    assert _looks_like_failure("1 failed, 4 passed") is True
    assert _looks_like_failure("FAILED tests/test_x.py::test_y") is True


def test_tester_run_exception_handling(tmp_path, monkeypatch):
    """Test tester run handles exceptions in the outer try block."""
    from furrow.agents.tester import TesterAgent

    async def fake_create_subprocess_exec(*args, **kwargs):
        raise RuntimeError("unexpected error")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    (tmp_path / "pyproject.toml").write_text("[project]\n")
    agent = TesterAgent(
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )

    # This should not raise, but return a TestResult with passed=False
    res = run(agent.run("goal", []))
    assert isinstance(res, TestResult)
    assert res.passed is False
    assert "RuntimeError" in res.summary
    assert "boom" in " ".join(res.failures) or "RuntimeError" in " ".join(res.failures)


def test_parse_test_result_invalid_json(tmp_path):
    """Test parse_test_result with invalid JSON falls back to heuristic."""
    from furrow.agents.tester import parse_test_result
    from furrow.config import TestResult

    # Invalid JSON but contains failure keywords
    result = parse_test_result("not json at all FAILED test_x")
    assert isinstance(result, TestResult)
    assert result.passed is False
    assert "FAILED" in result.summary

    # Valid JSON with passed true
    result = parse_test_result('{"passed": true, "summary": "all good", "failures": []}')
    assert result.passed is True

    # Valid JSON with passed false
    result = parse_test_result('{"passed": false, "summary": "failed", "failures": ["x"]}')
    assert result.passed is False
    assert result.failures == ["x"]
