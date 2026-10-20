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

    async def complete(self, prompt, system="", model=None):
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


def test_run_no_runner(tmp_path):
    agent = TesterAgent(
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )
    res = run(agent.run("goal", []))
    assert isinstance(res, TestResult)
    assert res.passed is True
    assert "no test runner" in res.summary.lower()


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


def test_run_failing_command(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("def test_fail():\n    assert False\n")
    client = FakeLLMClient(
        response='{"passed": false, "summary": "failed", "failures": ["assert False"]}'
    )
    agent = TesterAgent(
        client=client,
        settings=Settings(_env_file=None, workspace=tmp_path, request_timeout=30),
        workspace=tmp_path,
    )
    res = run(agent.run("goal", []))
    assert res.passed is False


def test_detect_tests_dir_no_config(tmp_path):
    """A tests/ directory with test files but no config file should still detect pytest."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_foo.py").write_text("def test_foo():\n    assert True\n")
    agent = TesterAgent(settings=Settings(_env_file=None, workspace=tmp_path), workspace=tmp_path)
    cmd = run(agent.detect_command())
    assert cmd is not None and "pytest" in cmd
