from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class TestTesterAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings(tester_model="test-model")
        return client

    @pytest.fixture
    def tasks(self):
        return [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="completed"),
        ]

    @pytest.mark.asyncio
    async def test_run_returns_passed_true_when_tests_pass(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)

        mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "All tests passed", "failures": []}')

        with patch.object(tester, '_run_tests', new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "10 passed, 0 failed"

            result = await tester.run("Test goal", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_returns_passed_false_when_tests_fail(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)

        mock_client.complete = AsyncMock(return_value='{"passed": false, "summary": "Some tests failed", "failures": ["test_foo failed"]}')

        with patch.object(tester, '_run_tests', new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "9 passed, 1 failed"

            result = await tester.run("Test goal", tasks)

        assert isinstance(result, TestResult)
        assert result.passed is False
        assert result.summary == "Some tests failed"
        assert result.failures == ["test_foo failed"]

    @pytest.mark.asyncio
    async def test_run_handles_test_runner_exception(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)

        mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')

        with patch.object(tester, '_run_tests', new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.side_effect = Exception("Test runner not found")

            result = await tester.run("Test goal", tasks)

        assert result.passed is False
        assert "Test runner not found" in result.summary
        assert "Test runner not found" in result.failures

    @pytest.mark.asyncio
    async def test_run_handles_invalid_json_from_llm_when_passed(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)

        mock_client.complete = AsyncMock(return_value='This response says passed but is not JSON')

        with patch.object(tester, '_run_tests', new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "tests output"

            result = await tester.run("Test goal", tasks)

        assert result.passed is True
        assert result.summary == 'This response says passed but is not JSON'
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_handles_invalid_json_from_llm_when_failed(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)

        mock_client.complete = AsyncMock(return_value='This response says failed but is not JSON')

        with patch.object(tester, '_run_tests', new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "tests output"

            result = await tester.run("Test goal", tasks)

        assert result.passed is False
        assert result.summary == 'This response says failed but is not JSON'
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_uses_correct_model(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)

        mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')

        with patch.object(tester, '_run_tests', new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "tests output"

            await tester.run("Test goal", tasks)

        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs.get('model') == "test-model"

    @pytest.mark.asyncio
    async def test_run_includes_goal_and_test_output_in_prompt(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)

        mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')

        with patch.object(tester, '_run_tests', new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "test output here"

            await tester.run("Specific test goal", tasks)

        call_args = mock_client.complete.call_args.args[0]
        assert "Specific test goal" in call_args
        assert "test output here" in call_args

    @pytest.mark.asyncio
    async def test_run_tests_returns_output_when_pytest_succeeds(self, mock_client):
        tester = TesterAgent(client=mock_client)

        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_subprocess:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"5 passed", b""))
            mock_subprocess.return_value = mock_proc

            output = await tester._run_tests()

        assert "5 passed" in output

    @pytest.mark.asyncio
    async def test_run_tests_tries_multiple_runners(self, mock_client):
        tester = TesterAgent(client=mock_client)

        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FileNotFoundError(f"Command not found: {args[0]}")
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"tests passed", b""))
            return mock_proc

        with patch('asyncio.create_subprocess_exec', new=mock_subprocess):
            output = await tester._run_tests()

        assert call_count == 3
        assert "tests passed" in output

    @pytest.mark.asyncio
    async def test_run_tests_returns_no_runner_found_when_all_fail(self, mock_client):
        tester = TesterAgent(client=mock_client)

        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_subprocess:
            mock_subprocess.side_effect = FileNotFoundError("Not found")

            output = await tester._run_tests()

        assert output == "No test runner found."