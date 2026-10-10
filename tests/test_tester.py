"""Tests for the TesterAgent's test-runner detection logic."""
import pytest

from furrow.config import Settings, Provider
from furrow.llm import LLMClient
from furrow.agents.tester import TesterAgent, _TEST_RUNNERS


@pytest.fixture
def tester(tmp_path):
    s = Settings(provider=Provider.OLLAMA, workspace=tmp_path)
    return TesterAgent(client=LLMClient(settings=s))


def test_detect_runners_finds_pytest(tester, tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest]")
    (tmp_path / "conftest.py").write_text("")
    result = tester._detect_test_runners(str(tmp_path))
    assert ["pytest", "-q"] in result
    # pytest should come before cargo since pytest manifest exists
    assert result.index(["pytest", "-q"]) < result.index(["cargo", "test", "-q"])


def test_detect_runners_finds_node_when_no_python(tester, tmp_path):
    (tmp_path / "package.json").write_text("{}")
    (tmp_path / "package-lock.json").write_text("{}")
    result = tester._detect_test_runners(str(tmp_path))
    # npm test should be found and preferred
    assert ["npm", "test", "--", "--silent"] in result


def test_detect_runners_all_fallback_when_no_manifests(tester, tmp_path):
    result = tester._detect_test_runners(str(tmp_path))
    # When no manifests, all runners are returned (fallback ordering)
    assert len(result) == len(_TEST_RUNNERS)


def test_runners_are_unique(tester, tmp_path):
    result = tester._detect_test_runners(str(tmp_path))
    # No duplicates
    seen = set()
    for cmd in result:
        key = tuple(cmd)
        assert key not in seen
        seen.add(key)
    assert len(seen) == len(_TEST_RUNNERS)