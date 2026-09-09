from pathlib import Path
from unittest.mock import Mock

from furrow.agents.tester import TesterAgent
from furrow.config import Settings


def test_resolve_workspace_returns_client_settings_workspace(tmp_path):
    client = Mock()
    client.settings = Settings(workspace=tmp_path)
    agent = TesterAgent(client=client)

    assert agent._resolve_workspace() == tmp_path


def test_resolve_workspace_falls_back_to_cwd():
    client = Mock()
    client.settings = Mock(workspace=None)
    agent = TesterAgent(client=client)

    assert agent._resolve_workspace() == Path.cwd()


def test_build_env_sets_pythonpath(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    agent = TesterAgent(client=Mock())

    env = agent._build_env(tmp_path)

    assert env["PYTHONPATH"] == str(tmp_path)


def test_build_env_prepends_to_existing_pythonpath(tmp_path, monkeypatch):
    existing_pythonpath = "/existing/python/path"
    monkeypatch.setenv("PYTHONPATH", existing_pythonpath)
    agent = TesterAgent(client=Mock())

    env = agent._build_env(tmp_path)

    assert env["PYTHONPATH"] == f"{tmp_path}:{existing_pythonpath}"
