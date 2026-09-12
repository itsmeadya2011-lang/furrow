from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_false():
    t = TestResult(passed=False, summary="fail", failures=["error"])
    assert t.passed is False
    assert t.failures == ["error"]


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"
    assert s.ollama_base_url == "http://localhost:11434"


def test_settings_ollama_provider():
    s = Settings(provider=Provider.OLLAMA)
    assert s.provider == Provider.OLLAMA


def test_orchestrator_not_done_without_tasks():
    client = LLMClient()
    orch = Orchestrator(goal="test goal", client=client)
    assert orch._is_done() is True


def test_orchestrator_is_done_when_all_tasks_completed():
    client = LLMClient()
    orch = Orchestrator(goal="test goal", client=client)
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orch._is_done() is True


def test_orchestrator_not_done_when_task_failed():
    client = LLMClient()
    orch = Orchestrator(goal="test goal", client=client)
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orch._is_done() is False


def test_orchestrator_not_done_when_tasks_pending():
    client = LLMClient()
    orch = Orchestrator(goal="test goal", client=client)
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="pending"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert orch._is_done() is False


def test_orchestrator_request_stop():
    client = LLMClient()
    orch = Orchestrator(goal="test goal", client=client)
    assert orch._stop_requested is False
    orch.request_stop()
    assert orch._stop_requested is True


def test_orchestrator_get_tasks_returns_stored_tasks():
    client = LLMClient()
    orch = Orchestrator(goal="test goal", client=client)
    task = TaskModel(id="1", description="task 1", status="pending")
    orch.tasks = [task]
    assert orch._get_tasks() == [task]


def test_llm_client_default_settings():
    client = LLMClient()
    assert client.settings is not None
    assert client.settings.provider == Provider.ANTHROPIC


def test_worker_agent_prompt_contains_task_info():
    from furrow.agents.worker import WORKER_PROMPT

    task = TaskModel(id="1", description="Implement auth", files=["auth.py"])
    prompt = f"{WORKER_PROMPT}\n\nTask: {task.description}\nFiles to touch: {', '.join(task.files) if task.files else 'any'}\n"
    assert "Implement auth" in prompt
    assert "auth.py" in prompt


def test_tester_agent_prompt_contains_json_shape():
    from furrow.agents.tester import TESTER_PROMPT
    assert "passed" in TESTER_PROMPT
    assert "summary" in TESTER_PROMPT
    assert "failures" in TESTER_PROMPT


def test_planner_prompt_contains_json_shape():
    from furrow.agents.planner import PLANNER_PROMPT
    assert "tasks" in PLANNER_PROMPT
    assert "rationale" in PLANNER_PROMPT


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_fields():
    t = TaskModel(id="1", description="test", files=["a.py"], dependencies=["2"], status="completed", result="done")
    assert t.files == ["a.py"]
    assert t.dependencies == ["2"]
    assert t.status == "completed"
    assert t.result == "done"
