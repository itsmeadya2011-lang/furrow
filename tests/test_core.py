import pytest
from furrow.config import Plan, Session, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_session_timestamp_update():
    s = Session()
    s.update_timestamp()
    assert s.created_at != ""
    assert s.updated_at != ""
    assert s.created_at == s.updated_at


def test_session_save_and_load(tmp_path):
    import json
    from furrow.config import Settings

    settings = Settings(workspace=tmp_path, session_file="test_session.json")
    session = Session(
        session_id="abc-123",
        goal="test goal",
        plan=Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok"),
        cycles=2,
    )
    settings.save_session(session)
    loaded = settings.load_session()
    assert loaded is not None
    assert loaded.session_id == "abc-123"
    assert loaded.goal == "test goal"
    assert loaded.cycles == 2
    assert loaded.plan is not None
    assert loaded.plan.tasks[0].id == "1"


def test_session_load_nonexistent(tmp_path):
    from furrow.config import Settings

    settings = Settings(workspace=tmp_path, session_file="nonexistent.json")
    loaded = settings.load_session()
    assert loaded is None


def test_orchestrator_initializes_with_empty_tasks():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    assert orch._is_done() is True
    assert orch._get_tasks() == []


def test_worker_prompt_is_json_instruction():
    from furrow.agents.prompts import WORKER_PROMPT
    assert "JSON" in WORKER_PROMPT
    assert "files" in WORKER_PROMPT
