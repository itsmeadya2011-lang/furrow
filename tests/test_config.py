"""Tests for configuration loading, furrow.json, and Settings."""

import json
from pathlib import Path

import pytest

from furrow.config import Provider, Settings


class TestSettingsEnv:
    def test_defaults(self) -> None:
        s = Settings()
        assert s.provider == Provider.ANTHROPIC
        assert s.max_cycles == 0
        assert s.max_parallel_tasks == 5

    def test_provider_enum(self) -> None:
        assert Provider("anthropic") == Provider.ANTHROPIC
        assert Provider("openai") == Provider.OPENAI
        assert Provider("ollama") == Provider.OLLAMA


class TestFurrowJson:
    """Tests for loading project-level furrow.json config."""

    def test_load_project_config_missing_file(self, tmp_path: Path) -> None:
        overrides = Settings.load_project_config(tmp_path)
        assert overrides == {}

    def test_load_project_config_valid(self, tmp_path: Path) -> None:
        config = {
            "$schema": "https://furrow.ai/schema/config.json",
            "name": "test-project",
            "settings": {
                "max_cycles": 10,
                "max_parallel_tasks": 3,
                "workspace": ".",
            },
        }
        (tmp_path / "furrow.json").write_text(json.dumps(config))
        overrides = Settings.load_project_config(tmp_path)
        assert overrides["max_cycles"] == 10
        assert overrides["max_parallel_tasks"] == 3

    def test_load_project_config_invalid_json(self, tmp_path: Path) -> None:
        (tmp_path / "furrow.json").write_text("{invalid json")
        overrides = Settings.load_project_config(tmp_path)
        assert overrides == {}

    def test_from_workspace_applies_overrides(self, tmp_path: Path) -> None:
        config = {
            "settings": {
                "max_cycles": 5,
            }
        }
        (tmp_path / "furrow.json").write_text(json.dumps(config))
        settings = Settings.from_workspace(tmp_path)
        assert settings.workspace == tmp_path
        assert settings.max_cycles == 5
