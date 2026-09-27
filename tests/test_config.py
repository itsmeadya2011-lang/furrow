from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_provider_enum():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"
    assert {Provider.ANTHROPIC, Provider.OPENAI, Provider.OLLAMA} == set(Provider)


def test_task_model_defaults():
    t = TaskModel(id="1", description="do a thing")
    assert t.id == "1"
    assert t.description == "do a thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_dependencies():
    t = TaskModel(id="2", description="b", files=["a.py"], dependencies=["1"], status="pending")
    assert t.dependencies == ["1"]
    assert t.files == ["a.py"]


def test_plan_model():
    tasks = [
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b", dependencies=["1"]),
    ]
    p = Plan(tasks=tasks, rationale="some rationale")
    assert p.rationale == "some rationale"
    assert p.tasks == tasks
    dumped = p.model_dump()
    assert dumped["rationale"] == "some rationale"
    assert dumped["tasks"][1]["dependencies"] == ["1"]


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.passed is True
    assert t.failures == []
    t2 = TestResult(passed=False, summary="bad", failures=["f1", "f2"])
    assert t2.passed is False
    assert t2.failures == ["f1", "f2"]


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.log_level == "INFO"
    assert s.model == "claude-sonnet-4-20250514"
