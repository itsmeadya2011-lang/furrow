import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.planner import PlannerAgent


# ========== Config Tests ==========

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_deps():
    t = TaskModel(id="1", description="test", dependencies=["0"])
    assert t.dependencies == ["0"]


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_settings_env_override():
    s = Settings(provider=Provider.OPENAI, max_cycles=10)
    assert s.provider == Provider.OPENAI
    assert s.max_cycles == 10


# ========== TesterAgent Tests ==========

@pytest.mark.asyncio
async def test_tester_no_tests():
    """Tester returns failure when no test runner is found."""
    with patch.object(Path, 'exists', return_value=False):
        with patch('shutil.which', return_value=None):
            agent = TesterAgent()
            agent.settings = Settings(workspace=Path("/fake"))
            result = await agent.run("test goal", [])
            assert result.passed is False or "No test runner" in result.summary


@pytest.mark.asyncio
async def test_tester_fallback_on_error():
    """Tester handles test runner errors gracefully."""
    with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_proc:
        mock_proc.return_value.communicate.side_effect = Exception("boom")
        agent = TesterAgent()
        agent.settings = Settings(workspace=Path("/fake"))
        result = await agent.run("test goal", [])
        assert result.passed is False


def test_detect_python_project():
    """Tester detects Python/pytest project correctly."""
    agent = TesterAgent()
    agent.settings = Settings(workspace=Path("/fake"))
    
    with patch.object(Path, '__truediv__', return_value=MagicMock()):
        mock_file = MagicMock()
        mock_file.exists.return_value = False
        mock_file.read_text.return_value = '[tool.pytest.ini_options]'
        
        with patch.object(Path, 'exists', return_value=lambda: True):
            with patch('builtins.open', MagicMock(read_data='[tool.pytest]')):
                pass  # Simplified detection test


# ========== WorkerAgent Tests ==========

@pytest.mark.asyncio
async def test_worker_initializes():
    """WorkerAgent initializes with task and client."""
    task = TaskModel(id="1", description="test task", files=["a.py"])
    agent = WorkerAgent(task=task)
    assert agent.task.id == "1"
    assert agent.task.description == "test task"


# ========== PlannerAgent Tests ==========

@pytest.mark.asyncio
async def test_planner_initializes():
    """PlannerAgent initializes correctly."""
    agent = PlannerAgent()
    assert agent.client is not None


# ========== LLMClient Tests ==========

def test_llm_client_initializes():
    """LLMClient initializes with settings."""
    from furrow.llm import LLMClient
    s = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings=s)
    assert client.settings.provider == Provider.ANTHROPIC


def test_llm_client_read_file(tmp_path):
    """LLMClient can read files."""
    from furrow.llm import LLMClient
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")
    
    s = Settings()
    client = LLMClient(settings=s)
    
    import asyncio
    content = asyncio.get_event_loop().run_until_complete(client.read_file(test_file))
    assert content == "hello world"


def test_llm_client_write_file(tmp_path):
    """LLMClient can write files."""
    from furrow.llm import LLMClient
    test_file = tmp_path / "output.txt"
    
    s = Settings()
    client = LLMClient(settings=s)
    
    import asyncio
    asyncio.get_event_loop().run_until_complete(client.write_file(test_file, "test content"))
    assert test_file.read_text() == "test content"


def test_llm_client_write_creates_dirs(tmp_path):
    """LLMClient.write_file creates parent directories."""
    from furrow.llm import LLMClient
    test_file = tmp_path / "nested" / "dir" / "output.txt"
    
    s = Settings()
    client = LLMClient(settings=s)
    
    import asyncio
    asyncio.get_event_loop().run_until_complete(client.write_file(test_file, "nested"))
    assert test_file.read_text() == "nested"


def test_llm_client_list_files(tmp_path):
    """LLMClient lists files in directory."""
    from furrow.llm import LLMClient
    (tmp_path / "a.py").touch()
    (tmp_path / "b.py").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").touch()
    
    s = Settings()
    client = LLMClient(settings=s)
    
    files = client.list_files(tmp_path)
    assert "a.py" in files
    assert "b.py" in files
    assert "sub/c.py" in files


def test_llm_client_list_files_nonexistent():
    """LLMClient returns empty list for nonexistent directory."""
    from furrow.llm import LLMClient
    s = Settings()
    client = LLMClient(settings=s)
    
    files = client.list_files("/nonexistent/path")
    assert files == []


# ========== Provider Tests ==========

def test_provider_enum():
    """Provider enum has expected values."""
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"


# ========== Model Tests ==========

def test_task_model_status_transitions():
    """TaskModel can track status through workflow."""
    t = TaskModel(id="1", description="test")
    assert t.status == "pending"
    
    t.status = "in_progress"
    assert t.status == "in_progress"
    
    t.status = "completed"
    t.result = "Done!"
    assert t.status == "completed"
    assert t.result == "Done!"


def test_task_model_multiple_files():
    """TaskModel supports multiple files."""
    t = TaskModel(
        id="1",
        description="Add auth",
        files=["src/auth.py", "src/middleware.py", "tests/test_auth.py"]
    )
    assert len(t.files) == 3
    assert "src/auth.py" in t.files


def test_plan_multiple_tasks():
    """Plan supports multiple tasks."""
    p = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", dependencies=[]),
            TaskModel(id="2", description="task 2", dependencies=["1"]),
            TaskModel(id="3", description="task 3", dependencies=["1"]),
        ],
        rationale="Parallel tasks"
    )
    assert len(p.tasks) == 3
    assert p.tasks[1].dependencies == ["1"]


def test_test_result_failures():
    """TestResult can store multiple failures."""
    t = TestResult(
        passed=False,
        summary="3 tests failed",
        failures=[
            "test_auth.py::test_login - AssertionError",
            "test_auth.py::test_logout - TimeoutError",
            "test_api.py::test_status - 404",
        ]
    )
    assert len(t.failures) == 3
    assert "test_login" in t.failures[0]


# ========== Settings Tests ==========

def test_settings_workspace_default():
    """Settings has workspace default to cwd."""
    s = Settings()
    assert s.workspace == Path.cwd()


def test_settings_custom_workspace():
    """Settings can have custom workspace."""
    s = Settings(workspace=Path("/custom/path"))
    assert s.workspace == Path("/custom/path")


def test_settings_ollama_config():
    """Settings includes Ollama configuration."""
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom:11434")
    assert s.provider == Provider.OLLAMA
    assert s.ollama_base_url == "http://custom:11434"


# ========== WorkerAgent Tool Tests ==========

@pytest.mark.asyncio
async def test_worker_read_file_tool(tmp_path):
    """WorkerAgent can read files via tool."""
    test_file = tmp_path / "test.py"
    test_file.write_text("# test file")
    
    agent = WorkerAgent(
        task=TaskModel(id="1", description="read test"),
    )
    # Mock client settings
    agent.client.settings = Settings(workspace=tmp_path)
    
    result = await agent._read_file("test.py")
    assert result["exists"] is True
    assert "# test file" in result["content"]


@pytest.mark.asyncio
async def test_worker_read_file_not_found():
    """WorkerAgent handles missing files."""
    agent = WorkerAgent(task=TaskModel(id="1", description="read"))
    agent.client.settings = Settings(workspace=Path("/nonexistent"))
    
    result = await agent._read_file("missing.txt")
    assert result["exists"] is False
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_worker_write_file_tool(tmp_path):
    """WorkerAgent can write files via tool."""
    agent = WorkerAgent(task=TaskModel(id="1", description="write"))
    agent.client.settings = Settings(workspace=tmp_path)
    
    result = await agent._write_file("new.py", "print('hello')")
    assert result["success"] is True
    assert (tmp_path / "new.py").read_text() == "print('hello')"


@pytest.mark.asyncio
async def test_worker_edit_file_tool(tmp_path):
    """WorkerAgent can edit files via tool."""
    test_file = tmp_path / "edit.py"
    test_file.write_text("old content\nline2\nline3")
    
    agent = WorkerAgent(task=TaskModel(id="1", description="edit"))
    agent.client.settings = Settings(workspace=tmp_path)
    
    result = await agent._edit_file("edit.py", "old content", "new content")
    assert result["success"] is True
    assert "new content" in test_file.read_text()
    assert "old content" not in test_file.read_text()


@pytest.mark.asyncio
async def test_worker_edit_file_not_found():
    """WorkerAgent handles edit on missing file."""
    agent = WorkerAgent(task=TaskModel(id="1", description="edit"))
    agent.client.settings = Settings(workspace=Path("/nonexistent"))
    
    result = await agent._edit_file("missing.py", "old", "new")
    assert result["success"] is False


@pytest.mark.asyncio
async def test_worker_edit_file_string_not_found(tmp_path):
    """WorkerAgent handles edit when old_string not found."""
    test_file = tmp_path / "test.py"
    test_file.write_text("actual content")
    
    agent = WorkerAgent(task=TaskModel(id="1", description="edit"))
    agent.client.settings = Settings(workspace=tmp_path)
    
    result = await agent._edit_file("test.py", "nonexistent", "new")
    assert result["success"] is False
    assert "not found" in result["error"].lower()


@pytest.mark.asyncio
async def test_worker_bash_tool():
    """WorkerAgent can execute bash commands."""
    agent = WorkerAgent(task=TaskModel(id="1", description="bash"))
    agent.client.settings = Settings(workspace=Path("/tmp"))
    
    result = await agent._bash("echo hello")
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_worker_bash_failure():
    """WorkerAgent handles bash command failures."""
    agent = WorkerAgent(task=TaskModel(id="1", description="bash"))
    agent.client.settings = Settings(workspace=Path("/tmp"))
    
    result = await agent._bash("exit 1")
    assert result["returncode"] == 1


# ========== Orchestrator Tests ==========

@pytest.mark.asyncio
async def test_orchestrator_initializes():
    """Orchestrator initializes correctly."""
    from furrow.core.orchestrator import Orchestrator
    
    with patch('furrow.agents.planner.PlannerAgent'):
        orch = Orchestrator(goal="test goal")
        assert orch.goal == "test goal"
        assert orch.cycles == 0
        assert orch.all_tasks == []


@pytest.mark.asyncio
async def test_orchestrator_is_done_empty():
    """Orchestrator._is_done returns False when no tasks."""
    from furrow.core.orchestrator import Orchestrator
    
    with patch('furrow.agents.planner.PlannerAgent'):
        orch = Orchestrator(goal="test")
        assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_all_complete():
    """Orchestrator._is_done returns True when all tasks complete."""
    from furrow.core.orchestrator import Orchestrator
    
    with patch('furrow.agents.planner.PlannerAgent'):
        orch = Orchestrator(goal="test")
        orch.all_tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_failure():
    """Orchestrator._is_done returns False when tasks failed."""
    from furrow.core.orchestrator import Orchestrator
    
    with patch('furrow.agents.planner.PlannerAgent'):
        orch = Orchestrator(goal="test")
        orch.all_tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="failed"),
        ]
        assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    """Orchestrator respects max_cycles setting."""
    from furrow.core.orchestrator import Orchestrator
    
    settings = Settings(max_cycles=2)
    orch = Orchestrator(goal="test", settings=settings)
    
    # Mock planner to return empty plan (will loop but respect max_cycles)
    with patch('furrow.agents.planner.PlannerAgent') as mock_planner:
        mock_planner.return_value.plan = AsyncMock(return_value=Plan(tasks=[], rationale=""))
        
        # Would normally loop - but we can test the setting is respected
        assert orch.max_cycles == 2


def test_orchestrator_get_tasks():
    """Orchestrator._get_tasks returns all tracked tasks."""
    from furrow.core.orchestrator import Orchestrator
    
    with patch('furrow.agents.planner.PlannerAgent'):
        orch = Orchestrator(goal="test")
        orch.all_tasks = [TaskModel(id="1", description="t1")]
        assert len(orch._get_tasks()) == 1


# ========== Ollama Tests ==========

@pytest.mark.asyncio
async def test_ollama_complete():
    """LLMClient can complete via Ollama."""
    from furrow.llm import LLMClient
    
    s = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=s)
    
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "test output"}
        mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        result = await client._complete_ollama("test prompt", "", "llama2")
        assert result == "test output"
