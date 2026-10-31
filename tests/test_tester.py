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


def test_parse_test_output_pytest_failed_lines():
    """pytest-style FAILED lines are extracted as structured failures."""
    from furrow.agents.tester import parse_test_output

    output = (
        "=================================== FAILURES ===================================\n"
        "FAILED tests/test_x.py::test_y - AssertionError: expected 1 got 2\n"
        "FAILED tests/test_a.py::test_b - ValueError: bad value\n"
        "=========================== 2 failed, 3 passed in 0.12s ===========================\n"
    )
    failures = parse_test_output(output)
    assert len(failures) == 2
    assert failures[0]["file"] == "tests/test_x.py"
    assert failures[0]["test"] == "test_y"
    assert "expected 1 got 2" in failures[0]["error"]
    assert failures[1]["file"] == "tests/test_a.py"
    assert failures[1]["test"] == "test_b"


def test_parse_test_output_deduplicates_by_location():
    """The same file:line location is not reported twice."""
    from furrow.agents.tester import parse_test_output

    output = (
        "FAILED tests/test_x.py::test_y - boom\n"
        "tests/test_x.py:12: some error\n"
    )
    failures = parse_test_output(output)
    # The file:line entry at line 12 should not duplicate the FAILED entry.
    assert len(failures) == 1


def test_parse_test_output_file_line_col():
    """file:line:col patterns are extracted as failures."""
    from furrow.agents.tester import parse_test_output

    output = "some error at src/foo.py:42:7\n"
    failures = parse_test_output(output)
    assert failures
    assert failures[0]["file"] == "src/foo.py"
    assert failures[0]["line"] == 42


def test_parse_test_output_file_line_without_col():
    """file:line patterns (without column) are extracted when no col pattern matched."""
    from furrow.agents.tester import parse_test_output

    output = "error at src/bar.py:99\n"
    failures = parse_test_output(output)
    assert failures
    assert failures[0]["file"] == "src/bar.py"
    assert failures[0]["line"] == 99


def test_parse_test_output_exception_fallback():
    """When no structured failures are found, exception types are used as a fallback."""
    from furrow.agents.tester import parse_test_output

    output = "RuntimeError: something broke\n"
    failures = parse_test_output(output)
    assert failures
    assert "RuntimeError" in failures[0]["error"]
    assert "something broke" in failures[0]["error"]


def test_parse_test_output_empty():
    """An empty output yields no failures."""
    from furrow.agents.tester import parse_test_output

    assert parse_test_output("") == []


def test_parse_test_output_attaches_exception_to_first_blank_error():
    """When structured failures exist but some have no error text, the first
    exception message is attached to the first blank-error failure."""
    from furrow.agents.tester import parse_test_output

    output = (
        "FAILED tests/test_x.py::test_y - \n"
        "ValueError: bad value\n"
    )
    failures = parse_test_output(output)
    assert failures
    assert any("ValueError" in (f.get("error") or "") for f in failures)
