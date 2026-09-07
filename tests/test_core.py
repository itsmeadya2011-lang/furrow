from furrow.config import FurrowState, Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []


def test_furrow_state_save_load(tmp_path):
    state = FurrowState(
        goal="build thing",
        cycle=1,
        tasks=[TaskModel(id="1", description="do thing")],
        completed=[],
        failed=[],
        last_plan=Plan(tasks=[], rationale="ok"),
    )
    path = tmp_path / "state.json"
    state.save(path)
    loaded = FurrowState.load(path)
    assert loaded.goal == state.goal
    assert loaded.cycle == state.cycle
    assert loaded.tasks[0].id == "1"
    assert loaded.last_plan.rationale == "ok"


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"
    assert list(Provider) == [Provider.ANTHROPIC, Provider.OPENAI, Provider.OLLAMA]


def test_settings_defaults():
    s = Settings()
    assert s.max_cycles == 20
    assert s.log_level == "INFO"
