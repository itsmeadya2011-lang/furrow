from furrow.config import Plan, Settings, TaskModel


def test_plan_without_rationale_defaults_to_empty_string():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")])
    assert plan.rationale == ""


def test_plan_with_rationale():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="test reason")
    assert plan.rationale == "test reason"


def test_settings_defaults():
    settings = Settings()
    assert settings.worker_max_rounds == 8
    assert settings.max_parallel_tasks == 5
    assert settings.provider.value == "anthropic"
    assert settings.max_cycles == 0


def test_settings_custom_values(tmp_path):
    settings = Settings(
        worker_max_rounds=12,
        max_parallel_tasks=10,
        workspace=tmp_path,
    )
    assert settings.worker_max_rounds == 12
    assert settings.max_parallel_tasks == 10
    assert settings.workspace == tmp_path