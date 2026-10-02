"""Tests for the WorkerAgent's file-change extraction and summary logic."""

from __future__ import annotations

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class _FakeClient:
    def __init__(self):
        from furrow.config import Settings

        self.settings = Settings()

    async def read_file(self, path):
        return ""

    async def write_file(self, path, content):
        pass


class TestExtractFileChanges:
    def test_no_file_blocks(self):
        worker = WorkerAgent(task=TaskModel(id="1", description="do"), client=_FakeClient())
        result = worker._extract_file_changes("No files here, just text.")
        assert result == []

    def test_single_file_block(self):
        response = "[[FILE: src/foo.py]]\nprint('hello')\n[[END]]\nDone!"
        worker = WorkerAgent(task=TaskModel(id="1", description="do"), client=_FakeClient())
        changes = worker._extract_file_changes(response)
        assert len(changes) == 1
        assert changes[0][0] == "src/foo.py"
        assert changes[0][1] == "print('hello')"

    def test_multiple_file_blocks(self):
        response = "[[FILE: a.py]]\ncontent_a\n[[END]]\n[[FILE: b.py]]\ncontent_b\n[[END]]"
        worker = WorkerAgent(task=TaskModel(id="1", description="do"), client=_FakeClient())
        changes = worker._extract_file_changes(response)
        assert len(changes) == 2
        assert changes[0][0] == "a.py"
        assert changes[1][0] == "b.py"

    def test_multiline_file_content(self):
        response = "[[FILE: main.py]]\ndef hello():\n    print('hi')\n[[END]]"
        worker = WorkerAgent(task=TaskModel(id="1", description="do"), client=_FakeClient())
        changes = worker._extract_file_changes(response)
        assert len(changes) == 1
        assert "def hello():" in changes[0][1]
        assert "    print('hi')" in changes[0][1]


class TestBuildSummary:
    def test_summary_with_changes(self):
        response = "[[FILE: a.py]]\ncode\n[[END]]\nSummary text here."
        worker = WorkerAgent(task=TaskModel(id="1", description="do"), client=_FakeClient())
        changes = worker._extract_file_changes(response)
        summary = worker._build_summary(changes, response)
        assert "Modified 1 file(s)" in summary
        assert "a.py" in summary
        assert "Summary text here." in summary

    def test_summary_without_changes(self):
        response = "Just some description, no files."
        worker = WorkerAgent(task=TaskModel(id="1", description="do"), client=_FakeClient())
        changes = worker._extract_file_changes(response)
        summary = worker._build_summary(changes, response)
        assert "No file changes detected" in summary
        assert "Just some description" in summary


class TestStripRichTags:
    def test_output_stripper_removes_tags(self):
        from furrow.core.orchestrator import _strip_rich

        assert _strip_rich("[bold green]Hello[/bold green]") == "Hello"
        assert _strip_rich("[yellow]Warning[/yellow]") == "Warning"
        assert _strip_rich("plain text") == "plain text"
