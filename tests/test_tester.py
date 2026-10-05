from furrow.agents.tester import TesterAgent
from furrow.config import Settings
from furrow.llm import LLMClient


def test_candidate_runners_python_project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("")
    monkeypatch.chdir(tmp_path)
    tester = TesterAgent(client=LLMClient(settings=Settings()), settings=Settings())
    runners = tester._candidate_runners()
    assert runners[0] == ["pytest", "-q"]
    assert ["python", "-m", "pytest", "-q"] in runners
    assert all("npm" not in c[0] for c in runners)


def test_candidate_runners_node_project(tmp_path, monkeypatch):
    (tmp_path / "package.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    tester = TesterAgent(client=LLMClient(settings=Settings()), settings=Settings())
    runners = tester._candidate_runners()
    assert runners[0] == ["npm", "test", "--", "--silent"]
    assert ["pnpm", "test", "--", "--silent"] in runners
    assert ["yarn", "test", "--silent"] in runners


def test_candidate_runners_cargo_project(tmp_path, monkeypatch):
    (tmp_path / "Cargo.toml").write_text("")
    monkeypatch.chdir(tmp_path)
    tester = TesterAgent(client=LLMClient(settings=Settings()), settings=Settings())
    assert tester._candidate_runners() == [["cargo", "test", "-q"]]


def test_candidate_runners_go_project(tmp_path, monkeypatch):
    (tmp_path / "go.mod").write_text("")
    monkeypatch.chdir(tmp_path)
    tester = TesterAgent(client=LLMClient(settings=Settings()), settings=Settings())
    assert tester._candidate_runners() == [["go", "test", "./..."]]


def test_candidate_runners_unknown_project_falls_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tester = TesterAgent(client=LLMClient(settings=Settings()), settings=Settings())
    runners = tester._candidate_runners()
    assert runners[0] == ["pytest", "-q"]
    assert ["cargo", "test", "-q"] in runners


def test_candidate_runners_multi_language(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "package.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    tester = TesterAgent(client=LLMClient(settings=Settings()), settings=Settings())
    runners = tester._candidate_runners()
    # python runners first, then node
    assert runners[0] == ["pytest", "-q"]
    assert ["npm", "test", "--", "--silent"] in runners
    index_pytest = runners.index(["python", "-m", "pytest", "-q"])
    index_npm = runners.index(["npm", "test", "--", "--silent"])
    assert index_pytest < index_npm
