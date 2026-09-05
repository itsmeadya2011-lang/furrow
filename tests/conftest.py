"""Configuration for pytest test suite."""
import pytest


@pytest.fixture(autouse=True)
def _configure_logging():
    """Ensure logging is configured for tests."""
    from furrow.logging import configure

    configure("WARNING")
    yield
