from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from furrow.agents.tester import MAX_FIX_ATTEMPTS, TesterAgent
from furrow.config import Settings


class FakeClient:
    """Deterministic fake LLM client: scripted responses, recorded writes."""

    def __init__(self, workspace: Path) -> None:
        self.settings = SimpleNamespace(workspace=workspace, tester_model="test-model")
        self.complete_response = ""
        self.complete_calls: list[tuple[str, str | None]] = []
        self.writes: list[tuple[str, str]] = []

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        self.complete_calls.append((prompt, model))
        return self.complete_response

    async def write_file(self, path: str | Path, content: str) -> None:
        self.writes.append((str(path), content))


def _patch_commands(
    monkeypatch: pytest.MonkeyPatch,
    agent: TesterAgent,
    outcomes: list[tuple[int, str]],
) -> None:
    """Replace ``_run_command`` with a fake that yields the given outcomes."""

    async def fake_run_command(
        self: TesterAgent, cmd: list[str], timeout: float = 120.0
    ) -> tuple[int, str]:
        return outcomes.pop(0)

    monkeypatch.setattr(TesterAgent, "_run_command", fake_run_command)


async def test_all_checks_pass_returns_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeClient(tmp_path)
    agent = TesterAgent(client=client)
    _patch_commands(monkeypatch, agent, [(0, ""), (0, ""), (0, "")])
    result = await agent.run("the goal", [])

    assert result.passed is True
    assert result.failures == []
    assert client.complete_calls == []


async def test_failed_check_produces_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeClient(tmp_path)
    agent = TesterAgent(client=client)
    _patch_commands(monkeypatch, agent, [(1, "tests failed\n"), (0, ""), (0, "")])
    result = await agent.run("the goal", [])

    assert result.passed is False
    assert len(result.failures) == 1
    assert "tests check failed" in result.failures[0]


async def test_missing_tool_is_skipped_not_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeClient(tmp_path)
    agent = TesterAgent(client=client)
    _patch_commands(monkeypatch, agent, [(1, "No module named pytest"), (0, ""), (0, "")])
    result = await agent.run("the goal", [])

    assert result.passed is True
    assert result.failures == []
    assert "tests: skipped" in result.summary


async def test_timeout_is_reported_as_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeClient(tmp_path)
    agent = TesterAgent(client=client)
    _patch_commands(
        monkeypatch,
        agent,
        [
            (124, "Timeout: command timed out after 120s: python -m pytest -q"),
            (0, ""),
            (0, ""),
        ],
    )
    result = await agent.run("the goal", [])

    assert result.passed is False
    assert any("timed out" in failure.lower() for failure in result.failures)


async def test_fix_loop_applies_the_patch_and_reruns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeClient(tmp_path)
    client.complete_response = (
        '{"summary": "fix tests", "files": {"src/app.py": "print(\'fixed\')\\n"}}'
    )
    agent = TesterAgent(client=client)
    _patch_commands(
        monkeypatch,
        agent,
        [
            (1, "tests failed\n"), (0, ""), (0, ""),  # initial run
            (0, ""), (0, ""), (0, ""),  # run after the patch
        ],
    )
    result = await agent.run("the goal", [])

    assert result.passed is True
    assert client.writes == [(str(tmp_path / "src/app.py"), "print('fixed')\n")]


async def test_fix_loop_gives_up_after_the_attempt_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeClient(tmp_path)
    client.complete_response = '{"summary": "nope", "files": {"src/app.py": "x\\n"}}'
    agent = TesterAgent(client=client)
    _patch_commands(
        monkeypatch,
        agent,
        [
            (1, "fail\n"), (0, ""), (0, ""),  # initial run
            (1, "fail\n"), (0, ""), (0, ""),  # rerun after attempt 1
            (1, "fail\n"), (0, ""), (0, ""),  # rerun after attempt 2
        ],
    )
    result = await agent.run("the goal", [])

    assert result.passed is False
    assert len(client.complete_calls) == MAX_FIX_ATTEMPTS
    assert len(client.writes) == MAX_FIX_ATTEMPTS


async def test_fix_loop_ignores_an_unparsable_patch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeClient(tmp_path)
    client.complete_response = "This is not JSON at all, just garbage."
    agent = TesterAgent(client=client)
    _patch_commands(monkeypatch, agent, [(1, "fail\n"), (0, ""), (0, "")])
    result = await agent.run("the goal", [])

    assert result.passed is False
    assert client.writes == []
    assert len(client.complete_calls) == 1


async def test_paths_escaping_the_workspace_are_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(workspace=tmp_path)
    client = FakeClient(tmp_path)
    client.complete_response = (
        '{"summary": "escape attempt", "files": {"../evil.py": "evil\\n", "safe.py": "safe\\n"}}'
    )
    agent = TesterAgent(client=client, settings=settings)
    _patch_commands(
        monkeypatch,
        agent,
        [
            (1, "fail\n"), (0, ""), (0, ""),  # initial run
            (0, ""), (0, ""), (0, ""),  # run after the patch
        ],
    )
    result = await agent.run("the goal", [])

    assert result.passed is True
    assert client.writes == [(str(tmp_path / "safe.py"), "safe\n")]
