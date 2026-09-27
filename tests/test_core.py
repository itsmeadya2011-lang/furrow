from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_with_dependencies():
    t = TaskModel(id="t2", description="dependent", dependencies=["t1"])
    assert t.dependencies == ["t1"]
    assert t.status == "pending"
    assert t.files == []


def test_plan_with_dependencies():
    tasks = [
        TaskModel(id="t1", description="first"),
        TaskModel(id="t2", description="second", dependencies=["t1"]),
    ]
    p = Plan(tasks=tasks, rationale="chain")
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["t1"]
    assert p.rationale == "chain"
