"""Unit tests for the pure helpers and local file operations in ``furrow.llm``.

Nothing here touches the network: only ``extract_json`` and the filesystem-backed
``read_file`` / ``write_file`` / ``list_files`` are exercised.
"""

from __future__ import annotations

import json
import os

import pytest

from furrow import config as config_module
from furrow.config import Settings
from furrow.llm import DEFAULT_MAX_TOKENS, LLMClient, extract_json

# --------------------------------------------------------------------------- #
# extract_json
# --------------------------------------------------------------------------- #


def test_extract_json_plain_object():
    assert extract_json('{"a": 1, "b": [2, 3]}') == {"a": 1, "b": [2, 3]}


def test_extract_json_ignores_surrounding_whitespace():
    assert extract_json('\n\n  {"ok": true}  \n') == {"ok": True}


def test_extract_json_json_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_bare_fence():
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_single_line_fence():
    assert extract_json('```{"a": 1}```') == {"a": 1}


def test_extract_json_unterminated_opening_fence():
    assert extract_json('```json\n{"a": 1}') == {"a": 1}


def test_extract_json_bare_array():
    assert extract_json('[{"id": "1"}, {"id": "2"}]') == [{"id": "1"}, {"id": "2"}]


def test_extract_json_fenced_array():
    assert extract_json('```json\n[1, 2]\n```') == [1, 2]


def test_extract_json_scalar():
    assert extract_json('"just a string"') == "just a string"


def test_extract_json_invalid_raises_value_error_not_jsondecodeerror():
    with pytest.raises(ValueError) as excinfo:
        extract_json("{definitely-not-json")
    # JSONDecodeError is a ValueError subclass, so the type check has to be explicit.
    assert not isinstance(excinfo.value, json.JSONDecodeError)
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_extract_json_invalid_message_previews_the_input():
    with pytest.raises(ValueError) as excinfo:
        extract_json("{definitely-not-json")
    assert "definitely-not-json" in str(excinfo.value)


def test_extract_json_invalid_message_previews_fenced_input_after_stripping():
    with pytest.raises(ValueError) as excinfo:
        extract_json('```json\n{"broken": \n```')
    assert "broken" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# LLMClient construction / settings resolution
# --------------------------------------------------------------------------- #


def test_client_without_arguments_uses_the_module_singleton():
    assert LLMClient().settings is config_module.settings


def test_client_with_explicit_settings_keeps_them(tmp_path):
    settings = Settings(_env_file=None, workspace=tmp_path)
    assert LLMClient(settings).settings is settings


def test_client_is_not_a_singleton_by_default():
    assert LLMClient() is not LLMClient()


def test_default_max_tokens():
    assert DEFAULT_MAX_TOKENS == 4096


def test_anthropic_property_without_a_key_raises_and_names_both_env_vars(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FURROW_ANTHROPIC_API_KEY", raising=False)
    client = LLMClient(Settings(_env_file=None, workspace=tmp_path, anthropic_api_key=None))
    with pytest.raises(ValueError) as excinfo:
        _ = client.anthropic
    message = str(excinfo.value)
    assert "FURROW_ANTHROPIC_API_KEY" in message
    assert "ANTHROPIC_API_KEY" in message


# --------------------------------------------------------------------------- #
# Local file helpers
# --------------------------------------------------------------------------- #


def _client(tmp_path) -> LLMClient:
    return LLMClient(Settings(_env_file=None, workspace=tmp_path))


async def test_write_file_creates_missing_parent_directories(tmp_path):
    client = _client(tmp_path)
    target = tmp_path / "deeply" / "nested" / "out.txt"
    assert not target.parent.exists()
    await client.write_file(target, "hello")
    assert target.is_file()
    assert target.read_text() == "hello"


async def test_write_file_accepts_str_paths(tmp_path):
    client = _client(tmp_path)
    await client.write_file(str(tmp_path / "as_str.txt"), "ok")
    assert (tmp_path / "as_str.txt").read_text() == "ok"


async def test_read_file_round_trip(tmp_path):
    client = _client(tmp_path)
    target = tmp_path / "notes.md"
    content = "line one\nline two\n"
    await client.write_file(target, content)
    assert await client.read_file(target) == content


async def test_list_files_returns_paths_relative_to_the_directory(tmp_path):
    client = _client(tmp_path)
    (tmp_path / "a.txt").write_text("a")
    nested = tmp_path / "pkg"
    nested.mkdir()
    (nested / "mod.py").write_text("x = 1")
    (nested / "sub").mkdir()
    (nested / "sub" / "deep.txt").write_text("d")

    listed = await client.list_files(tmp_path)

    assert set(listed) == {
        "a.txt",
        os.path.join("pkg", "mod.py"),
        os.path.join("pkg", "sub", "deep.txt"),
    }
    # rglob order is filesystem-dependent, so only membership is contractual.
    assert len(listed) == 3


async def test_list_files_missing_directory_returns_empty_list(tmp_path):
    client = _client(tmp_path)
    assert await client.list_files(tmp_path / "does" / "not" / "exist") == []


async def test_list_files_ignores_directories(tmp_path):
    client = _client(tmp_path)
    (tmp_path / "only_a_dir").mkdir()
    assert await client.list_files(tmp_path) == []
