
from furrow.agents.tester import TesterAgent
from furrow.config import TestResult


class FakeLLMClient:
    """Duck-typed LLMClient returning canned responses in order."""

    def __init__(self, responses, settings=None):
        from furrow.config import Settings

        self.responses = list(responses)
        self.settings = settings or Settings()
        self.prompts: list[str] = []

    async def complete(self, prompt, system="", model=None):
        self.prompts.append(prompt)
        return self.responses.pop(0) if self.responses else ""


async def test_test_result_parses_fenced_json(tmp_path):
    from furrow.config import Settings

    s = Settings(test_command='python3 -c "print(\'ok\')"', workspace=tmp_path)
    fake = FakeLLMClient(
        ['```json\n{"passed": true, "summary": "ok", "failures": []}\n```'],
        settings=s,
    )
    tester = TesterAgent(client=fake)
    result = await tester.run("goal", [])
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "ok"
    assert result.failures == []


async def test_no_test_runner_skips_llm(tmp_path):
    from furrow.config import Settings

    s = Settings(test_command="definitely-not-a-real-command-xyz", workspace=tmp_path)
    fake = FakeLLMClient(["should not be used"], settings=s)
    tester = TesterAgent(client=fake)
    result = await tester.run("goal", [])
    assert result.passed is True
    assert "skipping" in result.summary.lower()
    assert fake.prompts == []


async def test_custom_test_command(tmp_path):
    from furrow.config import Settings

    s = Settings(test_command='python3 -c "print(\'ok\')"', workspace=tmp_path)
    fake = FakeLLMClient([], settings=s)
    tester = TesterAgent(client=fake)
    output = await tester._run_tests()
    assert "ok" in output
    assert "exit 0" in output


async def test_non_json_response_heuristic(tmp_path):
    from furrow.config import Settings

    s = Settings(test_command='python3 -c "print(\'ok\')"', workspace=tmp_path)
    fake = FakeLLMClient(["All tests passed"], settings=s)
    tester = TesterAgent(client=fake)
    result = await tester.run("goal", [])
    assert result.passed is True