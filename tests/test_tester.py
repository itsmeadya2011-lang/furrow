import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.agents.tester import TesterAgent
from furrow.config import TaskModel, TestResult


class MockLLMClient:
    def __init__(self):
        self.settings = type("Settings", (), {"tester_model": "test-model"})()

    async def complete(self, prompt: str, model: str | None = None) -> str:
        return '{"passed": true, "summary": "All tests passed", "failures": []}'


@pytest.mark.asyncio
async def test_tester_initialization():
    client = MockLLMClient()
    tester = TesterAgent(client=client)
    assert tester.client is not None


@pytest.mark.asyncio
async def test_tester_run_success():
    client = MockLLMClient()
    tester = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="test")]

    with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "5 passed in 0.5s"
        result = await tester.run("goal", tasks)

    assert result.passed is True
    assert result.summary == "All tests passed"


@pytest.mark.asyncio
async def test_tester_run_failure():
    client = MockLLMClient()

    async def mock_complete(prompt: str, model: str | None = None) -> str:
        return '{"passed": false, "summary": "2 failed", "failures": ["test_a", "test_b"]}'

    client.complete = mock_complete
    tester = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="test")]

    with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "2 failed, 3 passed"
        result = await tester.run("goal", tasks)

    assert result.passed is False
    assert len(result.failures) == 2


@pytest.mark.asyncio
async def test_tester_run_exception():
    client = MockLLMClient()
    tester = TesterAgent(client=client)
    tasks = [TaskModel(id="1", description="test")]

    with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run:
        mock_run.side_effect = Exception("Test runner not found")
        result = await tester.run("goal", tasks)

    assert result.passed is False
    assert "Test runner not found" in result.summary


@pytest.mark.asyncio
async def test_parse_test_output():
    client = MockLLMClient()
    tester = TesterAgent(client=client)

    output = "5 passed, 2 failed, 1 error, 3 skipped in 1.23s"
    parsed = tester._parse_test_output(output)

    assert parsed["passed"] == 5
    assert parsed["failed"] == 2
    assert parsed["errors"] == 1
    assert parsed["skipped"] == 3


@pytest.mark.asyncio
async def test_parse_test_output_no_match():
    client = MockLLMClient()
    tester = TesterAgent(client=client)

    output = "Some other output"
    parsed = tester._parse_test_output(output)

    assert parsed["passed"] == 0
    assert parsed["failed"] == 0


@pytest.mark.asyncio
async def test_run_tests_no_runner():
    client = MockLLMClient()
    tester = TesterAgent(client=client)

    # All commands fail with FileNotFoundError
    with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
        result = await tester._run_tests()

    assert result == "No test runner found."