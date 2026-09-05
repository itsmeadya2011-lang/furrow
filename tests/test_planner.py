import pytest
from furrow.agents.planner import PlannerAgent


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param('{"key": "value", "num": 1}', {"key": "value", "num": 1}, id="plain-json"),
        pytest.param(
            '```json\n{"key": "value", "num": 1}\n```',
            {"key": "value", "num": 1},
            id="json-fence",
        ),
        pytest.param(
            '```\n{"key": "value", "num": 1}\n```',
            {"key": "value", "num": 1},
            id="generic-fence",
        ),
        pytest.param(
            '~~~\n{"key": "value", "num": 1}\n~~~',
            {"key": "value", "num": 1},
            id="tilde-fence",
        ),
        pytest.param(
            'Here is the plan: {"key": "value"} hope that helps',
            {"key": "value"},
            id="prose-embedded",
        ),
    ],
)
def test_extract_json_parses_valid_inputs(text, expected):
    assert PlannerAgent._extract_json(text) == expected


def test_extract_json_bad_json_raises_value_error():
    text = "This is not valid JSON {{"
    with pytest.raises(ValueError) as exc_info:
        PlannerAgent._extract_json(text)
    message = str(exc_info.value)
    assert "Preview" in message
    assert repr(text[:200]) in message
