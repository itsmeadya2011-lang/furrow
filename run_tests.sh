#!/bin/bash
cd /workspace/app
python -c "
import furrow
from furrow.config import Settings, TaskModel, Plan
from furrow.agents.parsing import parse_plan, parse_test_result
from furrow.llm import LLMClient
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import StateStore, RunState
from furrow.core.events import EventBus, Event
from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerAgent, is_command_denied
from furrow.agents.tester import TesterAgent
from furrow.web.server import app
print('All imports successful')
" 2>&1