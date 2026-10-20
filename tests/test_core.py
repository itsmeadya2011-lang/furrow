"""End-to-end smoke test of the plan -> execute -> test loop with fake models."""

from __future__ import annotations

from pathlib import Path

from furrow.config import Settings, TaskStatus
from furrow.core.events import Event
from furrow.core.orchestrator import Orchestrator
from tests.helpers import FakeLLMClient, make_plan, make_test_result, make_worker


async def test_full_loop_writes_files_and_reports_progress(
    settings: Settings, tmp_workspace: Path
) -> None:
    settings.max_cycles = 2
    plan_payload = make_plan("write the module")
    worker_payload = make_worker(
        summary="wrote module.py", edits=[{"path": "module.py", "content": "x = 1\n"}]
    )
    client = FakeLLMClient(
        settings,
        plans=[plan_payload, plan_payload],
        workers=[worker_payload, worker_payload],
        tests=[
            make_test_result(False, "boom", ["boom"]),
            make_test_result(False, "boom", ["boom"]),
        ],
    )
    events: list[Event] = []
    orch = Orchestrator("create a module", client=client, on_event=events.append)

    await orch.run()

    assert (tmp_workspace / "module.py").read_text(encoding="utf-8") == "x = 1\n"
    assert orch.cycles == 2
    assert len(orch.history) == 2
    assert orch.tasks[0].status == TaskStatus.COMPLETED
    assert orch.tasks[0].files_changed == ["module.py"]
    kinds = [event.kind for event in events]
    assert kinds.count("cycle_start") == 2
    assert kinds.count("test_result") == 2
    assert kinds[-1] == "done"
