import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import FILE_TOOLS


class FakeClient:
    """Stand-in LLMClient that records run_tool_loop calls without hitting an API."""

    def __init__(self) -> None:
        self.calls = []

    async def run_tool_loop(self, prompt, system, model, tools, max_rounds):
        self.calls.append((prompt, system, model, tools, max_rounds))
        return "Implemented the task."


@pytest.mark.asyncio
async def test_worker_invokes_tool_loop_with_file_tools():
    client = FakeClient()
    task = TaskModel(id="1", description="Add a foo function", files=["foo.py"])
    worker = WorkerAgent(task=task, client=client)

    result = await worker.run()

    assert result == "Implemented the task."
    assert len(client.calls) == 1
    prompt, _system, _model, tools, max_rounds = client.calls[0]
    assert "Add a foo function" in prompt
    assert "foo.py" in prompt
    assert tools == FILE_TOOLS
    assert max_rounds == 10
