from furrow.agents.tester import TesterAgent
from furrow.config import Provider, Settings, TestResult
from furrow.llm import LLMClient


def _make_agent() -> TesterAgent:
    s = Settings()
    s.provider = Provider.ANTHROPIC
    return TesterAgent(client=LLMClient(settings=s))


def test_tester_parse_valid_json():
    agent = _make_agent()
    result = agent._parse_result(
        '{"passed": true, "summary": "all good", "failures": []}', 0, ""
    )
    assert isinstance(result, TestResult)
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


def test_tester_parse_non_json_failure():
    """When the LLM returns garbage, fall back to the exit code (failure)."""
    agent = _make_agent()
    result = agent._parse_result("not json at all", 1, "some output")
    assert result.passed is False
    assert len(result.failures) == 1
    assert "code 1" in result.failures[0]


def test_tester_parse_non_json_success():
    agent = _make_agent()
    result = agent._parse_result("garbage but exit code zero", 0, "")
    assert result.passed is True
    assert result.failures == []


def test_tester_parse_json_missing_passed_falls_back():
    agent = _make_agent()
    result = agent._parse_result('{"summary": "oops"}', 1, "")
    assert result.passed is False
    assert len(result.failures) == 1
    assert "code 1" in result.failures[0]
