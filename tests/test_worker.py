"""Tests for the WorkerAgent file parsing logic."""

import pytest

from furrow.agents.worker import _parse_file_operations


class TestParseFileOperations:
    """Tests for _parse_file_operations which extracts file writes from LLM responses."""

    def test_parse_json_array(self) -> None:
        response = '[{"file": "src/main.py", "content": "print(1)"}, {"file": "test.py", "content": "pass"}]'
        ops = _parse_file_operations(response)
        assert len(ops) == 2
        assert ops[0] == ("src/main.py", "print(1)")
        assert ops[1] == ("test.py", "pass")

    def test_parse_json_with_summary(self) -> None:
        response = '[{"file": "a.py", "content": "x"}] Summary: wrote a file.'
        ops = _parse_file_operations(response)
        assert len(ops) == 1
        assert ops[0] == ("a.py", "x")

    def test_parse_json_wrapped_in_markdown(self) -> None:
        response = '```json\n[{"file": "a.py", "content": "hello"}]\n```'
        ops = _parse_file_operations(response)
        assert len(ops) == 1
        assert ops[0] == ("a.py", "hello")

    def test_parse_no_json_falls_back_to_markdown(self) -> None:
        response = """Here's the implementation:
```src/app.py
def main():
    pass
```
"""
        ops = _parse_file_operations(response)
        # Should find the code block with filename header
        assert len(ops) >= 1
        # At least one op should reference src/app.py
        file_paths = [op[0] for op in ops]
        assert any("app.py" in p for p in file_paths)

    def test_parse_empty_response(self) -> None:
        ops = _parse_file_operations("")
        assert ops == []

    def test_parse_text_without_blocks(self) -> None:
        response = "Here is a summary of what I plan to do. Nothing to write."
        ops = _parse_file_operations(response)
        assert ops == []
