from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_with_dependencies():
    t1 = TaskModel(id="1", description="first")
    t2 = TaskModel(id="2", description="second", dependencies=["1"])
    plan = Plan(tasks=[t1, t2], rationale="ordered work")
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.tasks[0].dependencies == []
    assert all(isinstance(dep, str) for dep in plan.tasks[1].dependencies)


def test_task_model_status_transitions():
    t = TaskModel(id="1", description="work")
    assert t.status == "pending"
    assert t.result is None

    t.status = "in_progress"
    assert t.status == "in_progress"

    t.status = "completed"
    t.result = "all good"
    assert t.status == "completed"
    assert t.result == "all good"

    t.status = "failed"
    t.result = "boom"
    assert t.status == "failed"
    assert t.result == "boom"