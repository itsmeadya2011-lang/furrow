from __future__ import annotations

from furrow.logging import configure


def pytest_configure() -> None:
    # Keep test output quiet and ensure structlog is in a known state.
    configure("ERROR")
