from furrow.agents import tester as tester_module
from furrow.agents.tester import TesterAgent
from furrow.config import Settings


class FakeProcess:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False

    async def communicate(self):
        return self._stdout, self._stderr

    async def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


class FakeClient:
    def __init__(self, settings, response=None):
        self.settings = settings
        self.response = (
            response
            or '{"passed": true, "summary": "ok", "failures": []}'
        )
        self.prompts = []

    async def complete(self, prompt, model=None):
        self.prompts.append(prompt)
        return self.response


async def test_tester_runs_commands_in_configured_workspace(monkeypatch, tmp_path):
    settings = Settings(workspace=tmp_path)
    calls = []

    async def create_subprocess_exec(*cmd, stdout=None, stderr=None, cwd=None):
        calls.append({"cmd": cmd, "cwd": cwd})
        return FakeProcess(stdout=b"ok\n")

    monkeypatch.setattr(
        tester_module.asyncio,
        "create_subprocess_exec",
        create_subprocess_exec,
    )
    client = FakeClient(settings)
    tester = TesterAgent(client=client)

    result = await tester.run("goal", [])

    assert calls == [{"cmd": ("pytest", "-q"), "cwd": tmp_path}]
    assert result.passed is True
    assert result.summary == "ok"


async def test_nonzero_exit_status_becomes_failed_test_result(
    monkeypatch, tmp_path
):
    settings = Settings(workspace=tmp_path)

    async def create_subprocess_exec(*cmd, stdout=None, stderr=None, cwd=None):
        return FakeProcess(
            returncode=7,
            stdout=b"failure output\n",
            stderr=b"err\n",
        )

    monkeypatch.setattr(
        tester_module.asyncio,
        "create_subprocess_exec",
        create_subprocess_exec,
    )
    tester = TesterAgent(settings=settings)

    result = await tester.run("goal", [])

    assert result.passed is False
    assert "status 7" in result.summary
    assert "failure output" in result.failures[-1]
    assert "err" in result.failures[-1]


async def test_test_command_exception_is_reported_as_failure(monkeypatch, tmp_path):
    settings = Settings(workspace=tmp_path)

    async def create_subprocess_exec(*cmd, stdout=None, stderr=None, cwd=None):
        raise RuntimeError("cannot start")

    monkeypatch.setattr(
        tester_module.asyncio,
        "create_subprocess_exec",
        create_subprocess_exec,
    )
    tester = TesterAgent(settings=settings)

    result = await tester.run("goal", [])

    assert result.passed is False
    assert "cannot start" in result.summary
    assert "cannot start" in result.failures[0]


async def test_run_tests_preserves_string_api(monkeypatch, tmp_path):
    settings = Settings(workspace=tmp_path)

    async def create_subprocess_exec(*cmd, stdout=None, stderr=None, cwd=None):
        return FakeProcess(stdout=b"output\n", stderr=b"error\n")

    monkeypatch.setattr(
        tester_module.asyncio,
        "create_subprocess_exec",
        create_subprocess_exec,
    )
    tester = TesterAgent(settings=settings)

    output = await tester._run_tests()

    assert output == "output\nerror\n"
    assert tester._last_test_execution is not None
    assert tester._last_test_execution.returncode == 0
