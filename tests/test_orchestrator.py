from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make sure the project root is importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from furrow.config import Settings  # noqa: E402
from furrow.core.orchestrator import Orchestrator  # noqa: E402


class StubLLMClient:
    """Deterministic LLM stand-in for orchestrator tests."""

    def __init__(self) -> None:
        self.settings = Settings()
        self.planner_calls: list[str] = []
        self.worker_calls = 0
        self._planner_response: str = json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "do it", "files": [], "dependencies": []}
                ],
                "rationale": "ok",
            }
        )
        self._worker_response: str = "All done."
        # When set, worker.complete() raises instead of returning text.
        self.worker_raises: Exception | None = None

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        # The planner prompt contains "Planning tasks" / "Plan" instructions; the
        # simplest reliable signal is whether the prompt mentions "tasks".
        # We treat the planner prompt as the one sent when the request has no
        # "Task:" header — workers get a "Task:" section.
        is_planner = "Task:" not in prompt
        if is_planner:
            self.planner_calls.append(prompt)
            return self._planner_response
        if self.worker_raises is not None:
            raise self.worker_raises
        self.worker_calls += 1
        return self._worker_response


@pytest.fixture
def tmp_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("furrow.config.settings", Settings(workspace=tmp_path))
    return tmp_path


@pytest.mark.asyncio
async def test_orchestrator_terminates_after_one_cycle(tmp_workspace: Path) -> None:
    client = StubLLMClient()
    orch = Orchestrator(goal="some goal", client=client)  # type: ignore[arg-type]
    await orch.run()
    # Planner ran exactly once because all tasks completed in cycle 1.
    assert len(client.planner_calls) == 1
    assert orch.cycles == 1
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles(tmp_workspace: Path) -> None:
    client = StubLLMClient()
    # Force the worker to fail so the planner keeps being called.
    client.worker_raises = RuntimeError("boom")

    # Configure the orchestrator's settings to allow only 2 cycles.
    from furrow.core import orchestrator as orch_module

    orch_module.settings = Settings(workspace=tmp_workspace, max_cycles=2)

    # Force the planner to always return a non-empty plan so the loop keeps
    # running unless bounded by max_cycles.
    client._planner_response = json.dumps(
        {
            "tasks": [
                {"id": "1", "description": "do it", "files": [], "dependencies": []}
            ],
            "rationale": "ok",
        }
    )
    orch = Orchestrator(goal="some goal", client=client)  # type: ignore[arg-type]
    await orch.run()
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_halts_on_two_empty_plans(tmp_workspace: Path) -> None:
    client = StubLLMClient()
    client._planner_response = json.dumps({"tasks": [], "rationale": "done"})

    # Allow many cycles so the only thing stopping us is the empty-plan guard.
    from furrow.core import orchestrator as orch_module

    orch_module.settings = Settings(workspace=tmp_workspace, max_cycles=100)

    orch = Orchestrator(goal="some goal", client=client)  # type: ignore[arg-type]
    await orch.run()
    # Two empty plans should halt before invoking a third.
    assert len(client.planner_calls) == 2