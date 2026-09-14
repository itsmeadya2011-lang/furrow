import sys
sys.path.insert(0, '/workspace/64b5f2dc-1bcb-407e-b26a-0216ac2acba8/sessions/agent_40bb51bc-cc02-498e-bc22-22c3bc858e2c')

# Test imports
from furrow.config import Plan, TaskModel, TestResult, FurrowState, Settings
from furrow.logging import setup_logging, get_logger
from furrow.llm import LLMClient
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.core.orchestrator import Orchestrator

print("All imports successful!")

# Test config models
task = TaskModel(id="1", description="test task", files=["test.py"])
plan = Plan(tasks=[task], rationale="test")
result = TestResult(passed=True, summary="ok", failures=[])

print(f"Task: {task.description}")
print(f"Plan: {plan.rationale}, {len(plan.tasks)} tasks")
print(f"TestResult: passed={result.passed}")

# Test state
state = FurrowState(goal="test goal")
state.add_cycle(1, "test goal", plan)
print(f"State: {state.current_cycle} cycles, {len(state.get_all_tasks())} tasks")

# Test settings
settings = Settings()
print(f"Settings: provider={settings.provider}, model={settings.model}")

print("\nAll tests passed!")