
from furrow.agents.parser import parse_file_edits


class TestParseFileEdits:
    def test_single_file_edit(self):
        response = '''```src/app/main.py
print("hello world")
```'''
        edits = parse_file_edits(response)
        assert len(edits) == 1
        assert edits[0][0] == "src/app/main.py"
        assert edits[0][1] == 'print("hello world")'

    def test_multiple_file_edits(self):
        response = '''```src/app/main.py
print("main")
```

```src/app/utils.py
def helper():
    return 42
```'''
        edits = parse_file_edits(response)
        assert len(edits) == 2
        assert edits[0][0] == "src/app/main.py"
        assert edits[0][1] == 'print("main")'
        assert edits[1][0] == "src/app/utils.py"
        assert edits[1][1] == "def helper():\n    return 42"

    def test_tilde_fences(self):
        response = '''~~~src/app/main.py
print("hello")
~~~'''
        edits = parse_file_edits(response)
        assert len(edits) == 1
        assert edits[0][0] == "src/app/main.py"
        assert edits[0][1] == 'print("hello")'

    def test_no_file_edits_plain_text(self):
        response = "This is just a plain text response with no code blocks."
        edits = parse_file_edits(response)
        assert edits == []

    def test_language_tagged_blocks_ignored(self):
        response = '''```python
def foo():
    pass
```

```src/app/main.py
print("real edit")
```'''
        edits = parse_file_edits(response)
        assert len(edits) == 1
        assert edits[0][0] == "src/app/main.py"
        assert edits[0][1] == 'print("real edit")'

    def test_dotfiles(self):
        response = '''```.gitignore
*.pyc
__pycache__/
```

```.env.example
FOO=bar
```'''
        edits = parse_file_edits(response)
        assert len(edits) == 2
        assert edits[0][0] == ".gitignore"
        assert edits[0][1] == "*.pyc\n__pycache__/"
        assert edits[1][0] == ".env.example"
        assert edits[1][1] == "FOO=bar"

    def test_empty_content_files(self):
        response = '''```src/empty.py
```
'''
        edits = parse_file_edits(response)
        assert len(edits) == 1
        assert edits[0][0] == "src/empty.py"
        assert edits[0][1] == ""

    def test_duplicate_paths_last_one_wins(self):
        response = '''```src/app/main.py
print("first")
```

```src/app/main.py
print("second")
```'''
        edits = parse_file_edits(response)
        assert len(edits) == 2
        assert edits[0][1] == 'print("first")'
        assert edits[1][1] == 'print("second")'

    def test_file_path_with_no_extension_but_slash(self):
        response = '''```src/config
key=value
```'''
        edits = parse_file_edits(response)
        assert len(edits) == 1
        assert edits[0][0] == "src/config"
        assert edits[0][1] == "key=value"

    def test_file_path_with_backslash(self):
        response = '''```src\\app\\main.py
print("windows path")
```'''
        edits = parse_file_edits(response)
        assert len(edits) == 1
        assert edits[0][0] == "src\\app\\main.py"
        assert edits[0][1] == 'print("windows path")'

    def test_indented_fenced_block(self):
        response = '''    ```src/app/main.py
    print("indented")
    ```'''
        edits = parse_file_edits(response)
        assert len(edits) == 1
        assert edits[0][0] == "src/app/main.py"
        assert edits[0][1] == 'print("indented")'

    def test_mixed_fence_lengths(self):
        response = '''`````src/app/main.py
print("five backticks")
`````

``````src/app/other.py
print("six backticks")
``````'''
        edits = parse_file_edits(response)
        assert len(edits) == 2
        assert edits[0][0] == "src/app/main.py"
        assert edits[1][0] == "src/app/other.py"

    def test_block_with_no_info_string_ignored(self):
        response = '''```
print("no path")
```

```src/app/main.py
print("has path")
```'''
        edits = parse_file_edits(response)
        assert len(edits) == 1
        assert edits[0][0] == "src/app/main.py"
        assert edits[0][1] == 'print("has path")'

    def test_filename_only_with_extension(self):
        response = '''```main.py
print("just filename")
```'''
        edits = parse_file_edits(response)
        assert len(edits) == 1
        assert edits[0][0] == "main.py"
        assert edits[0][1] == 'print("just filename")'

    def test_filename_without_extension_ignored(self):
        response = '''```README
# Header
```'''
        edits = parse_file_edits(response)
        assert edits == []

    def test_multiple_blocks_same_file_different_content(self):
        response = '''```src/app/main.py
line1
```

```src/app/main.py
line2
```

```src/app/main.py
line3
```'''
        edits = parse_file_edits(response)
        assert len(edits) == 3
        assert edits[0][1] == "line1"
        assert edits[1][1] == "line2"
        assert edits[2][1] == "line3"