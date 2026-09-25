from __future__ import annotations

from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from tests._fakes import FakeClient, make_settings

JSON_OK = '{"summary":"added foo","files":[{"path":"src/foo.py","content":"print(1)"}]}'


async def test_worker_writes_files_from_json(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(settings, responses=[JSON_OK])
    task = TaskModel(id="1", description="add foo module", files=["src/foo.py"])
    out = await WorkerAgent(task=task, client=client).run()
    assert out == "added foo"
    assert (tmp_path / "src" / "foo.py").read_text() == "print(1)"


async def test_worker_reads_existing_then_overwrites(tmp_path):
    settings = make_settings(tmp_path)
    (tmp_path / "bar.py").write_text("old content")
    captured = {}

    async def complete(prompt, system="", model=None):
        captured["prompt"] = prompt
        return '{"summary":"u","files":[{"path":"bar.py","content":"new content"}]}'

    client = FakeClient(settings, complete_fn=complete)
    task = TaskModel(id="1", description="update bar", files=["bar.py"])
    out = await WorkerAgent(task=task, client=client).run()
    assert out == "u"
    assert "old content" in captured["prompt"]
    assert (tmp_path / "bar.py").read_text() == "new content"


async def test_worker_non_json_returns_raw_and_writes_nothing(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(settings, responses=["totally not json"])
    task = TaskModel(id="1", description="do something", files=[])
    out = await WorkerAgent(task=task, client=client).run()
    assert out == "totally not json"
    # no files should have been created
    assert list((tmp_path).rglob("*")) == []


async def test_worker_fenced_json_is_accepted(tmp_path):
    settings = make_settings(tmp_path)
    fenced = "```json\n" + JSON_OK + "\n```"
    client = FakeClient(settings, responses=[fenced])
    task = TaskModel(id="1", description="add foo", files=["a.py"])
    out = await WorkerAgent(task=task, client=client).run()
    assert out == "added foo"
    assert (tmp_path / "src" / "foo.py").read_text() == "print(1)"


async def test_worker_no_files_context(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(settings, responses=[JSON_OK])
    task = TaskModel(id="1", description="add foo", files=[])  # no declared files
    await WorkerAgent(task=task, client=client).run()
    # file still written (worker returned it) even though it wasn't declared.
    assert (tmp_path / "src" / "foo.py").read_text() == "print(1)"


async def test_planner_parses_valid_plan(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(
        settings,
        responses=[
            '{"tasks":[{"id":"1","description":"x","files":["a.py"],"dependencies":[]}],"rationale":"r"}'
        ],
    )
    plan = await PlannerAgent(client=client).plan("goal")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "x"


async def test_planner_falls_back_on_invalid_json(tmp_path):
    settings = make_settings(tmp_path)
    client = FakeClient(settings, responses=["not json at all"])
    plan = await PlannerAgent(client=client).plan("goal")
    assert plan.tasks == []
    assert "failed" in plan.rationale.lower()
