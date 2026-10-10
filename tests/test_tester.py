import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from furrow.agents.tester import TesterAgent


class FakeLLMClient:
    def __init__(self, response: str, workspace):
        self.settings = SimpleNamespace(workspace=Path(workspace), tester_model="fake-model")
        self.response = response

    async def complete(self, prompt, system="", model=None):
        return self.response

    async def write_file(self, path, content):
        pass

    async def read_file(self, path):
        return ""

    def list_files(self, directory):
        return []


def make_tester(response: str, workspace) -> TesterAgent:
    return TesterAgent(client=FakeLLMClient(response, workspace))


@pytest.fixture(autouse=True)
def _stub_run_tests(monkeypatch):
    async def _fake(self):
        return ""

    monkeypatch.setattr(TesterAgent, "_run_tests", _fake)


async def test_test_candidates_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    tester = make_tester("", tmp_path)
    assert tester._test_candidates() == TesterAgent._pytest_candidates()


async def test_test_candidates_package_json(tmp_path):
    (tmp_path / "package.json").touch()
    tester = make_tester("", tmp_path)
    assert tester._test_candidates() == [
        ["npm", "test", "--", "--silent"],
        ["pnpm", "test", "--", "--silent"],
        ["yarn", "test", "--silent"],
    ]


async def test_test_candidates_cargo(tmp_path):
    (tmp_path / "Cargo.toml").touch()
    tester = make_tester("", tmp_path)
    assert tester._test_candidates() == [["cargo", "test", "-q"]]


async def test_test_candidates_go(tmp_path):
    (tmp_path / "go.mod").touch()
    tester = make_tester("", tmp_path)
    assert tester._test_candidates() == [["go", "test", "./..."]]


async def test_run_parses_fenced_json(tmp_path):
    payload = {"passed": True, "summary": "all good", "failures": []}
    response = "```json\n" + json.dumps(payload) + "\n```"
    tester = make_tester(response, tmp_path)

    result = await tester.run("goal", [])

    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


async def test_run_fallback_passed(tmp_path):
    tester = make_tester("all tests passed", tmp_path)
    result = await tester.run("goal", [])
    assert result.passed is True


async def test_run_fallback_failed(tmp_path):
    tester = make_tester("3 failures", tmp_path)
    result = await tester.run("goal", [])
    assert result.passed is False


async def test_run_handles_run_tests_exception(tmp_path, monkeypatch):
    async def boom(self):
        raise RuntimeError("no runner")

    monkeypatch.setattr(TesterAgent, "_run_tests", boom)
    tester = make_tester("ignored", tmp_path)

    result = await tester.run("goal", [])

    assert result.passed is False
    assert "no runner" in result.summary
    assert result.failures == ["no runner"]