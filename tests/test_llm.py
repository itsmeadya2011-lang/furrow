from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.llm import LLMClient


@pytest.mark.asyncio
async def test_run_shell_success():
    client = LLMClient()
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"hello stdout", b""))
    mock_proc.returncode = 0
    mock_proc.kill = MagicMock()

    with patch("furrow.llm.asyncio.create_subprocess_shell", return_value=mock_proc):
        code, stdout, stderr = await client.run_shell("echo hello")

    assert code == 0
    assert stdout == "hello stdout"
    assert stderr == ""


@pytest.mark.asyncio
async def test_run_shell_with_error():
    client = LLMClient()
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"command not found"))
    mock_proc.returncode = 127
    mock_proc.kill = MagicMock()

    with patch("furrow.llm.asyncio.create_subprocess_shell", return_value=mock_proc):
        code, stdout, stderr = await client.run_shell("badcommand")

    assert code == 127
    assert stderr == "command not found"


@pytest.mark.asyncio
async def test_run_shell_timeout():
    client = LLMClient()
    mock_proc = MagicMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_proc.kill = MagicMock()

    with patch("furrow.llm.asyncio.create_subprocess_shell", return_value=mock_proc):
        code, stdout, stderr = await client.run_shell("sleep 999", timeout=0.1)

    assert code == 124
    assert stderr == "Command timed out"
    mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_run_shell_exception():
    client = LLMClient()
    with patch("furrow.llm.asyncio.create_subprocess_shell", side_effect=OSError("boom")):
        code, stdout, stderr = await client.run_shell("anything")

    assert code == 127
    assert "boom" in stderr
