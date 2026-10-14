from unittest.mock import MagicMock

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Provider, Settings


@pytest.fixture
def mock_client(tmp_path):
    client = MagicMock()
    client.settings = Settings(
        provider=Provider.ANTHROPIC,
        workspace=tmp_path,
    )
    return client


class TestDetectTestCommands:
    @pytest.mark.asyncio
    async def test_detects_python_commands_for_pyproject_toml(self, mock_client, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        assert ["python", "-m", "pytest", "-q"] in commands
        assert ["pytest", "-q"] in commands

    @pytest.mark.asyncio
    async def test_detects_python_commands_for_setup_py(self, mock_client, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup\n")

        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        assert ["python", "-m", "pytest", "-q"] in commands

    @pytest.mark.asyncio
    async def test_detects_python_commands_for_setup_cfg(self, mock_client, tmp_path):
        (tmp_path / "setup.cfg").write_text("[metadata]\nname = test\n")

        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        assert ["python", "-m", "pytest", "-q"] in commands

    @pytest.mark.asyncio
    async def test_detects_python_commands_for_tox_ini(self, mock_client, tmp_path):
        (tmp_path / "tox.ini").write_text("[tox]\nenvlist = py310\n")

        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        assert ["python", "-m", "pytest", "-q"] in commands

    @pytest.mark.asyncio
    async def test_detects_python_commands_for_makefile(self, mock_client, tmp_path):
        (tmp_path / "Makefile").write_text("test:\n\tpytest\n")

        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        assert ["python", "-m", "pytest", "-q"] in commands

    @pytest.mark.asyncio
    async def test_detects_node_commands_for_package_json(self, mock_client, tmp_path):
        (tmp_path / "package.json").write_text('{"name": "test", "scripts": {"test": "jest"}}\n')

        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        assert ["npm", "test", "--", "--silent"] in commands
        assert ["pnpm", "test", "--", "--silent"] in commands
        assert ["yarn", "test", "--silent"] in commands

    @pytest.mark.asyncio
    async def test_detects_rust_commands_for_cargo_toml(self, mock_client, tmp_path):
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\nversion = "0.1.0"\n')

        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        assert ["cargo", "test", "-q"] in commands

    @pytest.mark.asyncio
    async def test_detects_go_commands_for_go_mod(self, mock_client, tmp_path):
        (tmp_path / "go.mod").write_text("module test\n\ngo 1.21\n")

        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        assert ["go", "test", "./..."] in commands

    @pytest.mark.asyncio
    async def test_detects_go_commands_for_go_sum(self, mock_client, tmp_path):
        (tmp_path / "go.sum").write_text("github.com/some/module v1.0.0\n")

        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        assert ["go", "test", "./..."] in commands

    @pytest.mark.asyncio
    async def test_fallback_when_no_manifest_found(self, mock_client, tmp_path):
        # Empty directory, no manifest files
        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        # Should have fallback commands for all languages
        assert ["pytest", "-q"] in commands
        assert ["python", "-m", "pytest", "-q"] in commands
        assert ["npm", "test", "--", "--silent"] in commands
        assert ["pnpm", "test", "--", "--silent"] in commands
        assert ["yarn", "test", "--silent"] in commands
        assert ["cargo", "test", "-q"] in commands
        assert ["go", "test", "./..."] in commands

    @pytest.mark.asyncio
    async def test_multiple_manifests_combines_commands(self, mock_client, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_path / "package.json").write_text('{"name": "test"}\n')
        (tmp_path / "Cargo.toml").write_text('[package]\nname = "test"\n')

        tester = TesterAgent(client=mock_client)
        commands = await tester._detect_test_commands()

        # Should have commands from all detected languages
        assert ["python", "-m", "pytest", "-q"] in commands
        assert ["npm", "test", "--", "--silent"] in commands
        assert ["cargo", "test", "-q"] in commands
