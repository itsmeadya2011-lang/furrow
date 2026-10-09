from __future__ import annotations

import httpx
import pytest

from furrow.config import Settings
from furrow.llm import LLMClient


class FakeTransportError(httpx.TransportError):
    pass


async def test_with_retry_respects_instance_max_retries():
    client = LLMClient(settings=Settings(max_retries=2))
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        raise FakeTransportError("boom")

    with pytest.raises(FakeTransportError):
        await client._with_retry(flaky)
    assert attempts == 2


async def test_with_retry_succeeds_after_transient_failure():
    client = LLMClient(settings=Settings(max_retries=3))
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise FakeTransportError("boom")
        return "ok"

    assert await client._with_retry(flaky) == "ok"
    assert attempts == 2


async def test_with_retry_does_not_retry_non_transient_errors():
    client = LLMClient(settings=Settings(max_retries=3))
    attempts = 0

    async def fail() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        await client._with_retry(fail)
    assert attempts == 1