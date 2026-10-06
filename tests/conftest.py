from __future__ import annotations

from pathlib import Path

import pytest

from furrow.config import Provider, Settings


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Path:
    """A temporary directory used as the Furrow workspace."""
    return tmp_path


@pytest.fixture
def mock_settings(temp_workspace: Path) -> Settings:
    """A Settings instance with test-friendly values."""
    return Settings(
        provider=Provider.ANTHROPIC,
        model="test-model",
        workspace=temp_workspace,
        max_cycles=0,
        max_parallel_tasks=5,
    )