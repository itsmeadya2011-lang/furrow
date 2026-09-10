import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    client.settings = MagicMock()
    client.settings.planner_model = "test-planner"
    client.settings.worker_model = "test-worker"
    client.settings.tester_model = "test-tester"
    return client


@pytest.fixture
def mock_planner(mock_llm_client):
    with patch("furrow.core.orchestrator.PlannerAgent") as mock:
        planner = AsyncMock()
        planner.plan = AsyncMock(return_value=Plan(
            tasks=[TaskModel(id="1", description="test task")],
            rationale="test rationale"
        ))
        mock.return_value = planner
        yield planner


@pytest.fixture
def mock_worker():
    with patch("furrow.core.orchestrator.WorkerAgent") as mock:
        worker = AsyncMock()
        worker.run = AsyncMock(return_value="Task completed successfully")
        mock.return_value = worker
        yield worker


@pytest.fixture
def mock_tester():
    with patch("furrow.core.orchestrator.TesterAgent") as mock:
        tester = AsyncMock()
        tester.run = AsyncMock(return_value=TestResult(
            passed=True,
            summary="All tests passed",
            failures=[]
        ))
        mock.return_value = tester
        yield tester


@pytest.mark.asyncio
async def test_orchestrator_runs_one_cycle(mock_llm_client, mock_planner, mock_worker, mock_tester):
    """Test that orchestrator runs a single cycle and completes."""
    orchestrator = Orchestrator(goal="Test goal", client=mock_llm_client)
    await orchestrator.run()
    
    # Verify planner was called
    mock_planner.plan.assert_called_once()
    
    # Verify worker was called
    mock_worker.run.assert_called_once()
    
    # Verify tester was called
    mock_tester.run.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrator_max_cycles(mock_llm_client, mock_planner, mock_worker, mock_tester):
    """Test that orchestrator respects max_cycles setting."""
    from furrow.config import settings
    original_max_cycles = settings.max_cycles
    settings.max_cycles = 2
    
    try:
        orchestrator = Orchestrator(goal="Test goal", client=mock_llm_client)
        await orchestrator.run()
        
        # Should run exactly 2 cycles
        assert mock_planner.plan.call_count == 2
    finally:
        settings.max_cycles = original_max_cycles


@pytest.mark.asyncio
async def test_orchestrator_tracks_tasks(mock_llm_client, mock_planner, mock_worker, mock_tester):
    """Test that orchestrator tracks tasks across cycles."""
    orchestrator = Orchestrator(goal="Test goal", client=mock_llm_client)
    await orchestrator.run()
    
    # Check that tasks are tracked
    assert len(orchestrator.all_tasks) == 1
    assert orchestrator.all_tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_orchestrator_context_building(mock_llm_client, mock_planner, mock_worker, mock_tester):
    """Test that orchestrator builds context from previous cycles."""
    # Make planner return different plans each cycle
    call_count = 0
    async def plan_side_effect(goal, context=""):
        nonlocal call_count
        call_count += 1
        return Plan(
            tasks=[TaskModel(id=str(call_count), description=f"task {call_count}")],
            rationale=f"rationale {call_count}"
        )
    mock_planner.plan.side_effect = plan_side_effect
    
    orchestrator = Orchestrator(goal="Test goal", client=mock_llm_client)
    await orchestrator.run()
    
    # Check that context was passed to planner on second call
    calls = mock_planner.plan.call_args_list
    assert len(calls) == 1  # Only one cycle since tests pass
    # Context should be empty on first call
    assert calls[0][0][1] == ""  # context arg


@pytest.mark.asyncio
async def test_orchestrator_handles_test_failure(mock_llm_client, mock_planner, mock_worker, mock_tester):
    """Test that orchestrator handles test failures and retries."""
    # Make tests fail on first cycle, pass on second
    call_count = 0
    async def test_side_effect(goal, tasks):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return TestResult(passed=False, summary="Tests failed", failures=["error 1"])
        return TestResult(passed=True, summary="All tests passed", failures=[])
    mock_tester.run.side_effect = test_side_effect
    
    orchestrator = Orchestrator(goal="Test goal", client=mock_llm_client)
    await orchestrator.run()
    
    # Should run 2 cycles
    assert mock_planner.plan.call_count == 2
    assert mock_tester.run.call_count == 2


@pytest.mark.asyncio
async def test_orchestrator_task_retry(mock_llm_client, mock_planner, mock_worker, mock_tester):
    """Test that orchestrator retries failed tasks."""
    # Make worker fail twice then succeed
    call_count = 0
    async def worker_side_effect():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("Simulated failure")
        return "Success on third try"
    mock_worker.run.side_effect = worker_side_effect
    
    orchestrator = Orchestrator(goal="Test goal", client=mock_llm_client)
    await orchestrator.run()
    
    # Should retry 3 times (max_retries = 3)
    assert mock_worker.run.call_count == 3
    assert orchestrator.all_tasks[0].status == "completed"