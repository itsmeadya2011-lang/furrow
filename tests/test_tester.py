from __future__ import annotations

import asyncio

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TestResult


class FakeLLMClient:
    def __init__(self, response="", raises=None):
        self.response = response
        self.raises = raises
        self.prompts = []
        self.models = []

    async def complete(self, prompt, system="", model=None, role="default"):
        self.prompts.append(prompt)
        self.models.append(model)
        if self.raises:
            raise self.raises
        return self.response


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

    agent = TesterAgent(
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=1),
        workspace=tmp_path,
    )
    res = run(agent.run("goal", []))
    assert isinstance(res, TestResult)
    assert res.passed is False
    assert "timed out" in res.summary.lower()
    assert killed["pg"] is True or killed["kill"] is True
