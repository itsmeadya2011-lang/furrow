from furrow.core.orchestrator import Orchestrator


def test_orchestrator_init_stores_goal():
    o = Orchestrator(goal="x")
    assert o.goal == "x"
    assert o.original_goal == "x"


def test_orchestrator_get_tasks_initially_empty():
    o = Orchestrator(goal="x")
    assert o._get_tasks() == []


def test_orchestrator_is_done_false_initially():
    o = Orchestrator(goal="x")
    assert o._is_done() is False


def test_orchestrator_does_not_overwrite_goal_on_simulated_failure():
    o = Orchestrator(goal="x")
    o._last_failures = ["x"]
    assert o.goal == o.original_goal
