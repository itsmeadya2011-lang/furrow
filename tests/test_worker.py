from furrow.agents.worker import WorkerAgent, _strip_code_fences


def test_parse_full_response():
    response = """SUMMARY: Created the auth module

FILES_CHANGED:
- src/auth.py
- tests/test_auth.py

FILE: src/auth.py
CONTENT:
def login():
    pass
ENDFILE

FILE: tests/test_auth.py
CONTENT:
def test_login():
    assert True
ENDFILE
"""
    parsed = WorkerAgent._parse_worker_response(response)
    assert parsed["summary"] == "Created the auth module"
    assert parsed["files_changed"] == ["src/auth.py", "tests/test_auth.py"]
    assert "src/auth.py" in parsed["file_blocks"]
    assert "tests/test_auth.py" in parsed["file_blocks"]
    assert "def login():" in parsed["file_blocks"]["src/auth.py"]


def test_parse_no_file_blocks():
    response = """SUMMARY: No changes needed

FILES_CHANGED:
(none)
"""
    parsed = WorkerAgent._parse_worker_response(response)
    assert parsed["summary"] == "No changes needed"
    assert parsed["files_changed"] == []
    assert parsed["file_blocks"] == {}


def test_parse_empty_response():
    parsed = WorkerAgent._parse_worker_response("")
    assert parsed["summary"] == ""
    assert parsed["files_changed"] == []
    assert parsed["file_blocks"] == {}


def test_parse_with_markdown_code_fences():
    response = """```
SUMMARY: Wrapped in fences

FILES_CHANGED:
- src/wrapped.py

FILE: src/wrapped.py
CONTENT:
print("hello")
ENDFILE
```"""
    parsed = WorkerAgent._parse_worker_response(response)
    assert parsed["summary"] == "Wrapped in fences"
    assert parsed["files_changed"] == ["src/wrapped.py"]
    assert "print(\"hello\")" in parsed["file_blocks"]["src/wrapped.py"]


def test_parse_strips_code_fences_from_file_content():
    response = """SUMMARY: Code block inside file

FILES_CHANGED:
- src/fenced.py

FILE: src/fenced.py
CONTENT:
```python
def foo():
    return 42
```
ENDFILE
"""
    parsed = WorkerAgent._parse_worker_response(response)
    content = parsed["file_blocks"]["src/fenced.py"]
    assert "def foo():" in content
    assert "return 42" in content
    # The ``` fences should be stripped
    assert "```" not in content


def test_parse_files_changed_single_line():
    response = """SUMMARY: One file

FILES_CHANGED:
- src/only.py
"""
    parsed = WorkerAgent._parse_worker_response(response)
    assert parsed["files_changed"] == ["src/only.py"]


def test_parse_files_changed_none_marker():
    response = """SUMMARY: Nothing

FILES_CHANGED:
(none)
"""
    parsed = WorkerAgent._parse_worker_response(response)
    assert parsed["files_changed"] == []


def test_strip_code_fences_helper():
    content = "```\nline1\nline2\n```"
    result = _strip_code_fences(content)
    assert result == "line1\nline2"


def test_strip_code_fences_no_fences():
    content = "plain content here"
    result = _strip_code_fences(content)
    assert result == "plain content here"


def test_strip_code_fences_empty():
    assert _strip_code_fences("") == ""