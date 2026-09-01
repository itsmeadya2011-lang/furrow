import pytest

from furrow.llm import LLMClient


@pytest.fixture
def client() -> LLMClient:
    return LLMClient()


async def test_read_file(client: LLMClient, tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("hello world")
    content = await client.read_file(p)
    assert content == "hello world"


async def test_read_file_accepts_string_path(client: LLMClient, tmp_path):
    p = tmp_path / "file.txt"
    p.write_text("via str")
    content = await client.read_file(str(p))
    assert content == "via str"


async def test_write_file_creates_file(client: LLMClient, tmp_path):
    p = tmp_path / "out.txt"
    await client.write_file(p, "written content")
    assert p.read_text() == "written content"


async def test_write_file_creates_parent_dirs(client: LLMClient, tmp_path):
    p = tmp_path / "nested" / "deeper" / "file.txt"
    await client.write_file(p, "deep content")
    assert p.exists()
    assert p.read_text() == "deep content"


async def test_write_file_overwrites_existing(client: LLMClient, tmp_path):
    p = tmp_path / "over.txt"
    p.write_text("old")
    await client.write_file(p, "new")
    assert p.read_text() == "new"


def test_list_files_recursive(client: LLMClient, tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    (tmp_path / "sub" / "nested").mkdir()
    (tmp_path / "sub" / "nested" / "c.py").write_text("c")

    files = client.list_files(tmp_path)
    assert "a.txt" in files
    assert "sub/b.txt" in files
    assert "sub/nested/c.py" in files


def test_list_files_empty_dir(client: LLMClient, tmp_path):
    assert client.list_files(tmp_path) == []


def test_list_files_nonexistent_dir(client: LLMClient, tmp_path):
    missing = tmp_path / "does_not_exist"
    assert client.list_files(missing) == []


def test_list_files_accepts_string(client: LLMClient, tmp_path):
    (tmp_path / "x.txt").write_text("x")
    files = client.list_files(str(tmp_path))
    assert "x.txt" in files