from __future__ import annotations

import asyncio

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.process import kill_process_tree


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_kill_process_tree_terminates_child():
    """kill_process_tree must terminate a spawned subprocess."""
    proc = await asyncio.create_subprocess_exec(
        "sleep", "2",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    assert proc.returncode is None  # still running
    try:
        await asyncio.wait_for(kill_process_tree(proc, new_session=True), timeout=10.0)
    except asyncio.TimeoutError:
        pytest.fail("kill_process_tree did not return within 10 seconds")
    # Give the OS a moment to reap the process.
    for _ in range(20):
        if proc.returncode is not None:
            break
        await asyncio.sleep(0.1)
    assert proc.returncode is not None, "process was not terminated"


@pytest.mark.asyncio
async def test_kill_process_tree_is_idempotent():
    """Calling kill_process_tree on an already-dead process must not raise."""
    proc = await asyncio.create_subprocess_exec(
        "true",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    await kill_process_tree(proc, new_session=True)  # must not raise


@pytest.mark.asyncio
async def test_kill_process_tree_no_new_session():
    """kill_process_tree with new_session=False still kills the direct child."""
    proc = await asyncio.create_subprocess_exec(
        "sleep", "30",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    assert proc.returncode is None
    await kill_process_tree(proc, new_session=False)
    await asyncio.sleep(0.2)
    assert proc.returncode is not None
