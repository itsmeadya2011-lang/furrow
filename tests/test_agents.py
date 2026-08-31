import json
from pathlib import Path

import pytest

from furrow.agents._json import _extract_json
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult


class FakeClient:
    def __init__(self, workspace: Path | None = None) -> None:
        self.settings = Settings(workspace=workspace or Path.cwd())
        self.files: dict[str, str] = {}
        self.calls: list[dict] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system, "model": model})
        return json.dumps({
            "tasks": [
                {"id": "1", "description": "do thing", "files": ["foo.py"], "dependencies": []}
            ],
            "rationale": "ok",
        })

    async def read_file(self, path: str | Path) -> str:
        p = Path(path)
        return p.read_text() if p.exists() else ""

    async def write_file(self, path: str | Path, content: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    def list_files(self, directory: str | Path) -> list[str]:
        base = Path(directory)
        if not base.exists():
            return []
        return [str(f.relative_to(base)) for f in base.rglob("*") if f.is_file()]


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_fenced():
    assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_fenced_no_lang():
    assert _extract_json('```\n{"a": 1}\n```') == '{"a": 1}'


@pytest.mark.asyncio
async def test_planner_parses_fenced_json(tmp_path: Path):
    client = FakeClient(tmp_path)
    agent = PlannerAgent(client=client)

    async def fake_complete(prompt: str, system: str = "", model: str | None = None) -> str:
        return "```json\n" + json.dumps({
            "tasks": [{"id": "1", "description": "x", "files": [], "dependencies": []}],
            "rationale": "ok",
        }) + "\n```"

    client.complete = fake_complete
    plan = await agent.plan("do something")
    assert isinstance(plan, Plan)
    assert plan.tasks[0].id == "1"


@pytest.mark.asyncio
async def test_worker_applies_edits(tmp_path: Path):
    client = FakeClient(tmp_path)
    task = TaskModel(id="1", description="create file", files=["hello.py"])
    agent = WorkerAgent(task=task, client=client, workspace=tmp_path)

    async def fake_complete(prompt: str, system: str = "", model: str | None = None) -> str:
        return json.dumps({
            "summary": "created hello.py",
            "edits": [{"path": "hello.py", "content": "print('hi')\n"}],
        })

    client.complete = fake_complete
    summary = await agent.run()
    assert summary == "created hello.py"
    assert (tmp_path / "hello.py").read_text() == "print('hi')\n"


@pytest.mark.asyncio
async def test_worker_deletes_file(tmp_path: Path):
    target = tmp_path / "old.py"
    target.write_text("old")
    client = FakeClient(tmp_path)
    task = TaskModel(id="1", description="remove file", files=["old.py"])
    agent = WorkerAgent(task=task, client=client, workspace=tmp_path)

    async def fake_complete(prompt: str, system: str = "", model: str | None = None) -> str:
        return json.dumps({
            "summary": "deleted old.py",
            "edits": [{"path": "old.py", "delete": True}],
        })

    client.complete = fake_complete
    summary = await agent.run()
    assert not target.exists()
    assert summary == "deleted old.py"


@pytest.mark.asyncio
async def test_worker_no_edits(tmp_path: Path):
    client = FakeClient(tmp_path)
    task = TaskModel(id="1", description="review only", files=[])
    agent = WorkerAgent(task=task, client=client, workspace=tmp_path)

    async def fake_complete(prompt: str, system: str = "", model: str | None = None) -> str:
        return json.dumps({"summary": "no changes needed", "edits": []})

    client.complete = fake_complete
    summary = await agent.run()
    assert summary == "no changes needed"
