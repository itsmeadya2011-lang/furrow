import pytest

from furrow.util import extract_json, strip_code_fences


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


def test_strip_code_fences():
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    # plain text unchanged
    assert strip_code_fences("hello world") == "hello world"
    # fenced without language
    assert strip_code_fences('```\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_plain():
    data = extract_json('{"tasks": [], "rationale": "ok"}')
    assert data == {"tasks": [], "rationale": "ok"}


def test_extract_json_fenced():
    data = extract_json('```json\n{"a": 1}\n```')
    assert data == {"a": 1}


def test_extract_json_with_prose():
    data = extract_json("Here you go:\n{\"a\": 1}\nHope that helps.")
    assert data == {"a": 1}


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json('{"a": 1} and {"b": 2}')
    with pytest.raises(ValueError):
        extract_json("[1, 2, 3]")