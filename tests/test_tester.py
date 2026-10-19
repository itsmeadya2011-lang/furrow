import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from furrow.agents.tester import TesterAgent, _extract_failures, _extract_json
from furrow.config import Settings


def test_json_extraction_code_block():
    text = '```json\n{"passed": true, "summary": "ok", "failures": []}\n```'
    assert json.loads(_extract_json(text))["passed"] is True


def test_failure_extraction():
    out = "FAILED test_x\nAssertionError: boom\n"
    fails = _extract_failures(out)
    assert any("FAILED" in f for f in fails)


def test_runner_fallback():
    s = Settings(_env_file=None, test_commands=[])
    client = MagicMock()
    client.complete = AsyncMock(return_value='{"passed": true, "summary": "ok", "failures": []}')
    client.settings = s
    t = TesterAgent(client=client, settings=s)
    # force no runners by patching _run_tests
    t._run_tests = AsyncMock(return_value="No test runner found or no output produced.")
    # complete returns passing json
    res = asyncio.get_event_loop().run_until_complete(t.run("g", []))
    assert res.passed is True
