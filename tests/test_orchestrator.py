import json
from unittest.mock import AsyncMock, patch

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def _plan_json() -> str:
    return json.dumps(
        {
            "tasks": [
                {"id": "1", "description": "first task"},
                {"id": "2", "description": "second task"},
            ],
            "rationale": "just do it",
        }
    )


def _test_result_json(passed: bool = True) -> str:
    return json.dumps(
        {
            "passed": passed,
            "summary": "all good" if passed else "stuff broke",
            "failures": [] if passed else ["test_x"],
        }
    )


def _make_client(passed: bool = True) -> LLMClient:
    client = LLMClient()
    plan_json = _plan_json()
    test_json = _test_result_json(passed)

    async def _complete(prompt: str, system: str = "", model: str | None = None) -> str:
        if model == settings.planner_model:
            return plan_json
        if model == settings.tester_model:
            return test_json
        return "worker output"

    client.complete = AsyncMock(side_effect=_complete)  # type: ignore[method-assign]
    return client


async def test_run_completes_when_tests_pass():
    client = _make_client(passed=True)
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value=""):
        orch = Orchestrator(goal="build it", client=client)
        await orch.run()

    assert orch.cycles == 1
    assert orch._is_done() is True


async def test_is_done_true_after_passing_cycle():
    client = _make_client(passed=True)
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value=""):
        orch = Orchestrator(goal="g", client=client)
        assert orch._is_done() is False
        await orch.run()
        assert orch._is_done() is True


async def test_cycles_increment():
    client = _make_client(passed=True)
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value=""):
        orch = Orchestrator(goal="g", client=client)
        assert orch.cycles == 0
        await orch.run()
        assert orch.cycles == 1


async def test_max_cycles_one_halts_after_one_cycle():
    # Workers fail so _is_done() stays False; only the max_cycles guard stops the loop.
    client = LLMClient()

    async def _complete(prompt: str, system: str = "", model: str | None = None) -> str:
        if model == settings.planner_model:
            return _plan_json()
        if model == settings.tester_model:
            return _test_result_json(passed=False)
        if model == settings.worker_model:
            raise RuntimeError("boom")
        return ""

    client.complete = AsyncMock(side_effect=_complete)  # type: ignore[method-assign]

    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value=""), \
         patch("furrow.core.orchestrator.settings", Settings(max_cycles=1)):
        orch = Orchestrator(goal="g", client=client)
        await orch.run()
        assert orch.cycles == 1


async def test_failed_tasks_tracked():
    client = LLMClient()

    async def _complete(prompt: str, system: str = "", model: str | None = None) -> str:
        if model == settings.planner_model:
            return _plan_json()
        if model == settings.tester_model:
            return _test_result_json(passed=False)
        if model == settings.worker_model:
            raise RuntimeError("boom")
        return ""

    client.complete = AsyncMock(side_effect=_complete)  # type: ignore[method-assign]

    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value=""), \
         patch("furrow.core.orchestrator.settings", Settings(max_cycles=1)):
        orch = Orchestrator(goal="g", client=client)
        await orch.run()

    assert orch._last_plan is not None
    tasks = orch._last_plan.tasks
    assert all(t.status == "failed" for t in tasks)
    assert all(t.result and "boom" in t.result for t in tasks)
    assert orch._is_done() is False
    assert orch.cycles == 1


async def test_orchestrator_uses_default_client():
    with patch.object(TesterAgent, "_run_tests", new_callable=AsyncMock, return_value=""):
        orch = Orchestrator(goal="g")
        assert isinstance(orch.client, LLMClient)

        # Empty plan -> _cycle returns early without invoking the tester.
        orch.client.complete = AsyncMock(  # type: ignore[method-assign]
            return_value=json.dumps({"tasks": [], "rationale": "nothing"})
        )
        await orch.run()

    assert orch.cycles == 1
    assert orch._last_plan is not None
    assert orch._last_plan.tasks == []