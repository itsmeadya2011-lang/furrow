from furrow.config import TaskModel


def test_task_model():
    task = TaskModel(id="1", description="Test task", files=["a.py", "b.py"])
    assert task.id == "1"
    assert task.description == "Test task"
    assert task.files == ["a.py", "b.py"]
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_dependencies():
    task = TaskModel(id="2", description="Dependent task", dependencies=["1"])
    assert task.dependencies == ["1"]