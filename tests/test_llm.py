import asyncio
from unittest.mock import patch

import pytest

from furrow.config import Provider, Settings
from furrow.llm import LLMClient


def test_instantiate_default():
    s = Settings(_env_file=None, anthropic_api_key="k")
    c = LLMClient(settings=s)
    assert c.settings is s


def test_complete_unsupported_provider():
    s = Settings(_env_file=None, provider="anthropic")
    # force unsupported by setting provider to something invalid via enum hack
    c = LLMClient(settings=s)
    # monkeypatch settings.provider to a value not handled
    c.settings.provider = "nope"
    with pytest.raises(ValueError):
        asyncio.get_event_loop().run_until_complete(c.complete("hi"))


def test_ollama_creates_client():
    s = Settings(_env_file=None, provider=Provider.OLLAMA, ollama_base_url="http://x:1")
    c = LLMClient(settings=s)
    client = c.ollama
    assert client is not None
    assert client.base_url.host == "x"


def test_retry_behavior():
    s = Settings(_env_file=None, provider=Provider.OLLAMA, max_retries=3, request_timeout=10)
    c = LLMClient(settings=s)
    calls = {"n": 0}

    async def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise asyncio.TimeoutError()
        return "ok"

    with patch.object(c, "_complete_ollama", side_effect=flaky):
        # bypass complete's dispatch by patching complete? Instead call retry-wrapped manually
        pass

    # Simpler: patch complete internals by calling c.complete with provider ollama
    # Re-implement: use tenacity retry on a mock method
    async def run():
        # call the actual complete; _complete_ollama is mocked to fail then succeed
        return await c.complete("hi")

    # reset
    calls["n"] = 0
    with patch.object(c, "_complete_ollama", side_effect=flaky):
        res = asyncio.get_event_loop().run_until_complete(run())
    assert res == "ok"
    assert calls["n"] == 2


def test_read_write_file(tmp_path):
    s = Settings(_env_file=None, provider=Provider.OLLAMA)
    c = LLMClient(settings=s)
    f = tmp_path / "a.txt"
    asyncio.get_event_loop().run_until_complete(c.write_file(str(f), "hello"))
    content = asyncio.get_event_loop().run_until_complete(c.read_file(str(f)))
    assert content == "hello"


def test_list_files(tmp_path):
    s = Settings(_env_file=None, provider=Provider.OLLAMA)
    c = LLMClient(settings=s)
    (tmp_path / "a.txt").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("y")
    files = c.list_files(tmp_path)
    assert "a.txt" in files
    assert "sub/b.txt" in files
