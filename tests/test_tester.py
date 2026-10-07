import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.llm import LLMClient


@pytest.fixture
def settings():
    return Settings(
        provider="anthropic",
        anthropic_api_key="test-key",
        tester_model="test-model",
    )


@pytest.fixture
def mock_client(settings):
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    client.complete = AsyncMock()
    return client


@pytest.fixture
def tasks():
    return [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_tests_returns_output_when_runner_found(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)
        with patch("furrow.agents.tester.asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"tests passed", b""))
            mock_proc.returncode = 0
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            output = await tester._run_tests()

            assert "tests passed" in output
            mock_exec.assert_called()

    @pytest.mark.asyncio
    async def test_run_tests_returns_no_runner_message_when_none_available(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)
        with patch("furrow.agents.tester.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError()

            output = await tester._run_tests()

            assert output == "No test runner found. No pytest, npm, pnpm, yarn, cargo, or go detected."

    @pytest.mark.asyncio
    async def test_run_tests_handles_non_zero_exit_code(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)
        with patch("furrow.agents.tester.asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"FAILED test_foo", b"error"))
            mock_proc.returncode = 1
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            output = await tester._run_tests()

            assert "FAILED test_foo" in output
            assert "[exit code: 1]" in output

    @pytest.mark.asyncio
    async def test_run_tests_handles_timeout(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)
        with patch("furrow.agents.tester.asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            with patch("furrow.agents.tester.asyncio.wait_for") as mock_wait:
                mock_wait.side_effect = [asyncio.TimeoutError()] * 7
                output = await tester._run_tests()

            assert output == "No test runner found. No pytest, npm, pnpm, yarn, cargo, or go detected."

    @pytest.mark.asyncio
    async def test_run_parses_llm_json_response(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)
        mock_client.complete.return_value = '{"passed": true, "summary": "All tests passed", "failures": []}'

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "pytest output"

            result = await tester.run("test goal", tasks)

            assert isinstance(result, TestResult)
            assert result.passed is True
            assert result.summary == "All tests passed"
            assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_handles_llm_non_json_response_fallback(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)
        mock_client.complete.return_value = "Tests passed successfully!"

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "pytest output"

            result = await tester.run("test goal", tasks)

            assert isinstance(result, TestResult)
            assert result.passed is True
            assert result.summary == "Tests passed successfully!"

    @pytest.mark.asyncio
    async def test_run_handles_llm_failed_response_fallback(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)
        mock_client.complete.return_value = "Tests failed: test_foo assertion error"

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "pytest output"

            result = await tester.run("test goal", tasks)

            assert isinstance(result, TestResult)
            assert result.passed is False
            assert result.summary == "Tests failed: test_foo assertion error"

    @pytest.mark.asyncio
    async def test_run_handles_json_with_failures(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)
        mock_client.complete.return_value = '{"passed": false, "summary": "Some tests failed", "failures": ["test_foo failed", "test_bar failed"]}'

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "pytest output"

            result = await tester.run("test goal", tasks)

            assert result.passed is False
            assert result.summary == "Some tests failed"
            assert len(result.failures) == 2

    @pytest.mark.asyncio
    async def test_run_uses_tester_model_from_settings(self, mock_client, tasks, settings):
        tester = TesterAgent(client=mock_client)
        mock_client.complete.return_value = '{"passed": true, "summary": "ok", "failures": []}'

        with patch.object(tester, "_run_tests", new_callable=AsyncMock):
            await tester.run("test goal", tasks)

        mock_client.complete.assert_called_once()
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs.get("model") == settings.tester_model

    @pytest.mark.asyncio
    async def test_run_passes_goal_and_test_output_to_llm(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)
        mock_client.complete.return_value = '{"passed": true, "summary": "ok", "failures": []}'

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.return_value = "test output here"
            await tester.run("my goal", tasks)

        call_args = mock_client.complete.call_args
        prompt = call_args[0][0]
        assert "my goal" in prompt
        assert "test output here" in prompt

    @pytest.mark.asyncio
    async def test_run_handles_exception_in_run_tests(self, mock_client, tasks):
        tester = TesterAgent(client=mock_client)
        mock_client.complete.return_value = '{"passed": true, "summary": "ok", "failures": []}'

        with patch.object(tester, "_run_tests", new_callable=AsyncMock) as mock_run_tests:
            mock_run_tests.side_effect = Exception("runner crashed")

            result = await tester.run("test goal", tasks)

            assert result.passed is False
            assert "runner crashed" in result.summary
            assert "runner crashed" in result.failures[0]


class TestTesterAgentInitialization:
    def test_initializes_with_default_client(self):
        with patch("furrow.agents.tester.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.settings = Settings()
            mock_client_class.return_value = mock_client
            tester = TesterAgent()
            assert tester.client is mock_client

    def test_initializes_with_provided_client(self, mock_client):
        tester = TesterAgent(client=mock_client)
        assert tester.client is mock_client

    def test_initializes_with_provided_settings(self):
        settings = Settings(provider="anthropic", anthropic_api_key="test")
        with patch("furrow.agents.tester.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.settings = settings
            mock_client_class.return_value = mock_client
            tester = TesterAgent(settings=settings)
            assert tester.client is mock_client
            mock_client_class.assert_called_once_with(settings=settings)