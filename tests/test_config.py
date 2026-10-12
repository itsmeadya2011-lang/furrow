"""Tests for configuration and LLM client improvements."""
from __future__ import annotations

from furrow.config import Provider, Settings, configure_logging


def test_default_max_tokens():
    """Settings should default max_tokens to 4096."""
    s = Settings()
    assert s.max_tokens == 4096


def test_custom_max_tokens():
    """Settings should accept a custom max_tokens value."""
    s = Settings(max_tokens=8192)
    assert s.max_tokens == 8192


def test_ollama_base_url_default():
    """Settings should default ollama_base_url."""
    s = Settings()
    assert s.ollama_base_url == "http://localhost:11434"


def test_provider_ollama_in_enum():
    """Provider enum should include OLLAMA."""
    assert Provider.OLLAMA == "ollama"


def test_configure_logging_runs():
    """configure_logging should execute without error."""
    configure_logging("INFO")
    configure_logging("DEBUG")
