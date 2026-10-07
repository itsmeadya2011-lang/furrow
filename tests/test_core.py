import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.context import should_compact


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


@pytest.mark.asyncio
async def test_should_compact():
    # Empty history should not compact
    assert not await should_compact([])
    
    # Small history should not compact
    assert not await should_compact([{"cycle": 1, "summary": "test"}])
    
    # Large history should compact
    history = [{"cycle": i, "summary": f"cycle {i}"} for i in range(15)]
    assert await should_compact(history, threshold=10)
    
    # Custom threshold
    assert not await should_compact([{"cycle": i, "summary": f"cycle {i}"} for i in range(5)], threshold=10)
    assert await should_compact([{"cycle": i, "summary": f"cycle {i}"} for i in range(5)], threshold=3)