from __future__ import annotations

from furrow.agents.tester import parse_test_output


def test_pytest_style_failed_line():
    output = "FAILED tests/test_foo.py::test_bar - AssertionError: boom"
    result = parse_test_output(output)
    assert len(result) == 1
    assert result[0]["file"] == "tests/test_foo.py"
    assert result[0]["test"] == "test_bar"
    assert result[0]["error"] == "AssertionError: boom"
    assert result[0]["line"] is None


def test_multiple_pytest_failures():
    output = """FAILED tests/test_a.py::test_one - AssertionError: first
FAILED tests/test_b.py::test_two - ValueError: second
FAILED tests/test_c.py::test_three - RuntimeError: third"""
    result = parse_test_output(output)
    assert len(result) == 3
    assert result[0]["file"] == "tests/test_a.py"
    assert result[0]["test"] == "test_one"
    assert result[0]["error"] == "AssertionError: first"
    assert result[1]["file"] == "tests/test_b.py"
    assert result[1]["test"] == "test_two"
    assert result[1]["error"] == "ValueError: second"
    assert result[2]["file"] == "tests/test_c.py"
    assert result[2]["test"] == "test_three"
    assert result[2]["error"] == "RuntimeError: third"


def test_file_line_col_pattern():
    output = "src/mod.py:42:5"
    result = parse_test_output(output)
    assert len(result) == 1
    assert result[0]["file"] == "src/mod.py"
    assert result[0]["line"] == 42
    assert result[0]["test"] is None
    assert result[0]["error"] == ""


def test_file_line_pattern_without_col():
    output = "tests/test_x.py:10"
    result = parse_test_output(output)
    assert len(result) == 1
    assert result[0]["file"] == "tests/test_x.py"
    assert result[0]["line"] == 10
    assert result[0]["test"] is None
    assert result[0]["error"] == ""


def test_exception_patterns():
    output = "ValueError: bad value\nRuntimeError: something went wrong"
    result = parse_test_output(output)
    assert len(result) == 2
    assert result[0]["error"] == "ValueError: bad value"
    assert result[1]["error"] == "RuntimeError: something went wrong"
    assert result[0]["file"] is None
    assert result[0]["line"] is None
    assert result[0]["test"] is None


def test_exception_patterns_not_extracted_when_failures_exist():
    output = """FAILED tests/test_foo.py::test_bar - AssertionError: boom
ValueError: this should not be separate"""
    result = parse_test_output(output)
    assert len(result) == 1
    assert result[0]["file"] == "tests/test_foo.py"
    assert result[0]["test"] == "test_bar"
    assert result[0]["error"] == "AssertionError: boom"


def test_no_failures_in_output():
    output = "5 passed, 0 failed in 0.12s\nAll tests passed!"
    result = parse_test_output(output)
    assert result == []


def test_empty_string_input():
    result = parse_test_output("")
    assert result == []


def test_duplicate_locations_deduplicated():
    output = """FAILED tests/test_foo.py::test_bar - AssertionError: boom
FAILED tests/test_foo.py::test_bar - AssertionError: boom again"""
    result = parse_test_output(output)
    assert len(result) == 1


def test_output_with_braces_inside_strings():
    output = 'FAILED tests/test_foo.py::test_bar - AssertionError: expected "foo {bar}" got "baz"'
    result = parse_test_output(output)
    assert len(result) == 1
    assert result[0]["error"] == 'AssertionError: expected "foo {bar}" got "baz"'


def test_multiple_file_line_col_and_file_line_patterns():
    output = """src/mod.py:42:5
tests/test_x.py:10
src/other.py:100:20"""
    result = parse_test_output(output)
    # The order may vary based on regex finditer
    assert len(result) == 3
    files = {r["file"] for r in result}
    assert files == {"src/mod.py", "tests/test_x.py", "src/other.py"}
    lines = {r["line"] for r in result}
    assert lines == {42, 10, 100}


def test_exception_pattern_with_no_message_after_colon():
    # The regex requires something after the colon, so this won't match
    output = "ValueError:"
    result = parse_test_output(output)
    assert result == []


def test_output_with_windows_line_endings():
    output = (
        "FAILED tests/test_foo.py::test_bar - AssertionError: boom\r\n"
        "FAILED tests/test_baz.py::test_qux - ValueError: oops"
    )
    result = parse_test_output(output)
    assert len(result) == 2
    assert result[0]["file"] == "tests/test_foo.py"
    assert result[0]["error"] == "AssertionError: boom"
    assert result[1]["file"] == "tests/test_baz.py"
    assert result[1]["error"] == "ValueError: oops"


def test_very_long_output():
    output = "\n".join([f"FAILED tests/test_{i}.py::test_{i} - Error: {i}" for i in range(100)])
    result = parse_test_output(output)
    assert len(result) == 100
    assert result[0]["file"] == "tests/test_0.py"
    assert result[99]["file"] == "tests/test_99.py"


def test_exception_fills_empty_error_fields_when_failures_exist():
    # Exception appears before FAILED line, so FAILED line is last
    # and its error is empty. The exception filling logic should fill it.
    output = """ValueError: fill me in
FAILED tests/test_foo.py::test_bar -"""
    result = parse_test_output(output)
    assert len(result) == 1
    assert result[0]["file"] == "tests/test_foo.py"
    assert result[0]["test"] == "test_bar"
    assert result[0]["error"] == "ValueError: fill me in"


def test_file_line_col_does_not_match_duplicate_file_line():
    output = """src/mod.py:42:5
src/mod.py:42"""
    result = parse_test_output(output)
    # The file:line pattern should not add a duplicate since file:line:col was already added
    assert len(result) == 1
    assert result[0]["file"] == "src/mod.py"
    assert result[0]["line"] == 42


def test_pytest_failure_with_file_line_col_combined():
    output = """FAILED tests/test_foo.py::test_bar - AssertionError: boom
tests/test_foo.py:42:5"""
    result = parse_test_output(output)
    # These are treated as different locations (test name vs line number)
    # so both are included
    assert len(result) == 2
    assert result[0]["file"] == "tests/test_foo.py"
    assert result[0]["test"] == "test_bar"
    assert result[0]["error"] == "AssertionError: boom"
    assert result[1]["file"] == "tests/test_foo.py"
    assert result[1]["line"] == 42
    assert result[1]["error"] == "AssertionError: boom"


def test_exception_only_no_failures():
    output = "TypeError: invalid type\nAttributeError: missing attr"
    result = parse_test_output(output)
    assert len(result) == 2
    assert result[0]["error"] == "TypeError: invalid type"
    assert result[1]["error"] == "AttributeError: missing attr"


def test_file_line_pattern_windows_path():
    # Windows paths with backslashes: the regex [^\s:]+\.py matches from the
    # last colon-separated segment ending in .py
    output = r"C:\path\to\file.py:10"
    result = parse_test_output(output)
    assert len(result) == 1
    assert result[0]["file"] == r"\path\to\file.py"
    assert result[0]["line"] == 10


def test_complex_mixed_output():
    output = """=================================== FAILURES ===================================
FAILED tests/test_foo.py::test_bar - AssertionError: expected 1 got 2
tests/test_foo.py:42:5
FAILED tests/test_baz.py::test_qux - ValueError: bad input
src/mod.py:100
RuntimeError: unexpected crash
=========================== 2 failed, 3 passed in 0.12s ==========================="""
    result = parse_test_output(output)
    assert len(result) == 4
    files = [f["file"] for f in result if f["file"]]
    assert "tests/test_foo.py" in files
    assert "tests/test_baz.py" in files
    assert "src/mod.py" in files
    tests = [f["test"] for f in result if f["test"]]
    assert "test_bar" in tests
    assert "test_qux" in tests
