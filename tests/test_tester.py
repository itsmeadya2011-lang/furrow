from __future__ import annotations

from pathlib import Path

from furrow.agents.tester import detect_test_commands


def test_detect_pytest_first(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("")
    cmds = detect_test_commands(tmp_path)
    assert cmds[0] == ["pytest", "-q"]


def test_detect_npm(tmp_path: Path):
    (tmp_path / "package.json").write_text("{}")
    cmds = detect_test_commands(tmp_path)
    assert ["npm", "test", "--", "--silent"] in cmds


def test_detect_fallback(tmp_path: Path):
    cmds = detect_test_commands(tmp_path)
    assert any(cmd[0] == "pytest" for cmd in cmds)