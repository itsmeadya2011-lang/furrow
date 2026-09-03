import asyncio
import tempfile
from pathlib import Path

from furrow.llm import LLMClient


def test_list_files_filters_pycache_and_git():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "a.py").touch()
        (p / "b.txt").touch()
        (p / "__pycache__").mkdir()
        (p / "__pycache__" / "x.py").touch()
        (p / ".git").mkdir()
        (p / ".git" / "HEAD").touch()

        files = LLMClient().list_files(tmpdir)
        assert "a.py" in files
        assert "b.txt" in files
        assert "__pycache__/x.py" not in files
        assert ".git/HEAD" not in files
