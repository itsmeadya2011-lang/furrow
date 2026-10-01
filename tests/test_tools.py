from pathlib import Path

from furrow.tools import ToolRegistry


async def test_tools_property_names():
    reg = ToolRegistry(workspace=Path("."))
    names = [t.name for t in reg.tools]
    assert names == ["read_file", "write_file", "list_files", "run_bash"]


async def test_write_and_read(tmp_path):
    reg = ToolRegistry(workspace=tmp_path)
    out = await reg.execute("write_file", {"path": "a/b.txt", "content": "hello"})
    assert out == "Wrote a/b.txt (5 bytes)"
    content = await reg.execute("read_file", {"path": "a/b.txt"})
    assert content == "hello"
    assert (tmp_path / "a" / "b.txt").read_text() == "hello"


async def test_read_missing(tmp_path):
    reg = ToolRegistry(workspace=tmp_path)
    out = await reg.execute("read_file", {"path": "missing.txt"})
    assert out.startswith("Error:")


async def test_list_files(tmp_path):
    reg = ToolRegistry(workspace=tmp_path)
    await reg.write_file("one.txt", "1")
    await reg.write_file("two.txt", "2")
    listing = await reg.list_files(".", recursive=False)
    names = listing.splitlines()
    assert "one.txt" in names and "two.txt" in names


async def test_list_files_recursive(tmp_path):
    reg = ToolRegistry(workspace=tmp_path)
    await reg.write_file("sub/deep.txt", "x")
    listing = await reg.execute("list_files", {"path": ".", "recursive": True})
    assert "sub" in listing


async def test_run_bash(tmp_path):
    reg = ToolRegistry(workspace=tmp_path)
    out = await reg.run_bash("echo bash_works")
    assert "bash_works" in out


async def test_unknown_tool(tmp_path):
    reg = ToolRegistry(workspace=tmp_path)
    out = await reg.execute("nope", {})
    assert "unknown tool" in out and "nope" in out
