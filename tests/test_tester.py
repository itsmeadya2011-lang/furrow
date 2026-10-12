"""Tests for the TesterAgent response parsing."""

from __future__ import annotations

import pytest
from furrow.agents.tester import TesterAgent
from furrow.config import Settings, TaskModel


class _FakeSettings(Settings):
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None


class FakeClient:
    def __init__(self, response: str = "") -> None:
        self._response = response
        self.settings = _FakeSettings()

    async def complete(self, prompt: str, model: str | None = None) -> str:
        return self._response


def test_parse_valid_json():
    client = FakeClient('{"passed": true, "summary": "All good", "failures": []}')
    agent = TesterAgent(client=client)
    result = agent._parse_response(client._response, "")
    assert result.passed is True
    assert result.summary == "All good"
    assert result.failures == []


def test_parse_json_with_failures():
    client = FakeClient(
        '{"passed": false, "summary": "2 failures", "failures": ["test_a failed", "test_b failed"]}'
    )
    agent = TesterAgent(client=client)
    result = agent._parse_response(client._response, "")
    assert result.passed is False
    assert len(result.failures) == 2


def test_parse_json_strips_markdown_fence():
    client = FakeClient(
        '```json\n{"passed": true, "summary": "ok", "failures": []}\n```'
    )
    agent = TesterAgent(client=client)
    result = agent._parse_response(client._response, "")
    assert result.passed is True


def test_parse_json_extracts_object_from_text():
    client = FakeClient(
        'Here is my analysis: {"passed": false, "summary": "broken", "failures": ["oops"]} done'
    )
    agent = TesterAgent(client=client)
    result = agent._parse_response(client._response, "")
    assert result.passed is False
    assert result.summary == "broken"
    assert result.failures == ["oops"]


def test_parse_fallback_string_match():
    client = FakeClient("The tests all passed successfully.")
    agent = TesterAgent(client=client)
    result = agent._parse_response(client._response, "")
    assert result.passed is True


def test_parse_fallback_failure():
    client = FakeClient("Tests failed with errors.")
    agent = TesterAgent(client=client)
    result = agent._parse_response(client._response, "")
    assert result.passed is False


def test_parse_empty_response():
    client = FakeClient("")
    agent = TesterAgent(client=client)
    result = agent._parse_response(client._response, "")
    assert result.passed is False