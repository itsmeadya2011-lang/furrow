from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_settings() -> None:
    from furrow.config import settings
    settings.max_cycles = 0