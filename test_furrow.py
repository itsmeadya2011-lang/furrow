python3 -c "
import sys
sys.path.insert(0, '.')
from furrow.config import Settings, Plan, TaskModel, TestResult, configure_logging
from furrow.llm import LLMClient
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent
from furrow.agents.tester import TesterAgent
from furrow.core.orchestrator import Orchestrator
print('✓ All imports successful')
settings = Settings()
print(f'✓ Settings loaded: provider={settings.provider}, model={settings.model}')
plan = Plan(tasks=[TaskModel(id='1', description='test task', files=['test.py'])], rationale='test')
print(f'✓ Plan creation works: {len(plan.tasks)} tasks')
result = TestResult(passed=True, summary='All good', failures=[])
print(f'✓ TestResult works: passed={result.passed}')
configure_logging('DEBUG')
print('✓ Logging configured')
print('✓ All basic tests passed!')
"