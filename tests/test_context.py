from unittest.mock import AsyncMock

import pytest

from furrow.config import Settings
from furrow.context import ContextCompactor


class MockLLMClient:
    def __init__(self):
        self.settings = Settings()
        self.complete = AsyncMock(return_value="Compacted summary")


def test_context_compactor_should_compact():
    compactor = ContextCompactor(client=MockLLMClient())
    assert compactor.should_compact("x" * 60000) is True
    assert compactor.should_compact("x" * 1000) is False


@pytest.mark.asyncio
async def test_context_compactor_compact():
    mock_client = MockLLMClient()
    compactor = ContextCompactor(client=mock_client)
    
    context = "x" * 60000
    goal = "test goal"
    
    result = await compactor.compact(context, goal)
    
    assert result == "Compacted summary"
    mock_client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_context_compactor_compact_state():
    mock_client = MockLLMClient()
    compactor = ContextCompactor(client=mock_client)
    compactor.max_context_chars = 1000  # Lower threshold for test
    
    # Mock state object
    class MockCycle:
        def __init__(self, cycle_num):
            self.cycle = cycle_num
            self.timestamp = "2024-01-01T00:00:00"
            self.plan_rationale = "test rationale"
            self.tasks = [
                {"id": f"{cycle_num}-{i}", "status": "completed", "description": "task " + "x" * 100, "result": "done"}
                for i in range(3)
            ]
            self.test_result = {"passed": True, "summary": "passed", "failures": []}
    
    class MockState:
        def __init__(self):
            self.cycles = [MockCycle(i) for i in range(5)]
    
    state = MockState()
    goal = "test goal"
    
    await compactor.compact_state(state, goal)
    
    # Should have called complete
    mock_client.complete.assert_called_once()