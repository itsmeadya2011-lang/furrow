from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from furrow.agents.tester import TesterAgent
from tests._fakes import FakeClient, make_settings


async def test_tester_parses_pass_result(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(settings, responses=['{"passed":true,"summary":"all green","failures":[]}'])
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="all green output")  # type: ignore[assignment]
    result = await tester.run("goal", [])
    assert result.passed is True
    assert result.summary == "all green"
    assert result.failures == []


async def test_tester_parses_fail_result(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(
        settings, responses=['{"passed":false,"summary":"broken","failures":["test_x failed"]}']
    )
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="some failing output")  # type: ignore[assignment]
    result = await tester.run("goal", [])
    assert result.passed is False
    assert "test_x failed" in result.failures


async def test_tester_fallback_on_plain_text(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(settings, responses=["looks like tests passed yay"])
    tester = TesterAgent(client=client)
    tester._run_tests = AsyncMock(return_value="...output...")  # type: ignore[assignment]
    result = await tester.run("goal", [])
    assert result.passed is True
    assert result.failures == []


async def test_tester_runs_in_workspace_cwd(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    client = FakeClient(settings)

    class FakeProc:
        async def communicate(self):
            return (b"out", b"err")

        def kill(self):
            pass

        async def wait(self):
            pass

    captured = {}

    async def fake_exec(*cmd, **kw):
        captured.update(kw)
        return FakeProc()

    import furrow.agents.tester as tester_mod

    monkeypatch.setattr(tester_mod.asyncio, "create_subprocess_exec", fake_exec)

    tester = TesterAgent(client=client)
    output = await tester._run_tests()
    assert Path(captured["cwd"]) == tmp_path
    assert "out" in output


async def test_tester_no_runner_returns_message(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    client = FakeClient(settings)

    async def fake_exec(*cmd, **kw):
        raise FileNotFoundError("no such runner")

    import furrow.agents.tester as tester_mod

    monkeypatch.setattr(tester_mod.asyncio, "create_subprocess_exec", fake_exec)

    tester = TesterAgent(client=client)
    output = await tester._run_tests()
    assert output == "No test runner found."


async def test_tester_timeout_kills_and_continues(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, test_timeout=0.1)
    client = FakeClient(settings)

    class HangingProc:
        async def communicate(self):
            await asyncio.sleep(10)

        def kill(self):
            pass

        async def wait(self):
            return 0

    calls = {"count": 0}

    async def fake_exec(*cmd, **kw):
        calls["count"] += 1
        return HangingProc()

    import furrow.agents.tester as tester_mod

    monkeypatch.setattr(tester_mod.asyncio, "create_subprocess_exec", fake_exec)

    tester = TesterAgent(client=client)
    output = await tester._run_tests()
    # All candidates timed out → no runner found message.
    assert output == "No test runner found."
    assert calls["count"] == len(tester_mod._TEST_CANDIDATES)
