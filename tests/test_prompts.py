from furrow.agents import prompts
from furrow.agents.prompts import (
    PLANNER_PROMPT,
    TESTER_PROMPT,
    WORKER_PROMPT,
    __all__,
)


def test_planner_prompt_exists():
    assert isinstance(PLANNER_PROMPT, str)
    assert len(PLANNER_PROMPT) > 0


def test_worker_prompt_exists():
    assert isinstance(WORKER_PROMPT, str)
    assert len(WORKER_PROMPT) > 0


def test_tester_prompt_exists():
    assert isinstance(TESTER_PROMPT, str)
    assert len(TESTER_PROMPT) > 0


def test_worker_prompt_has_tools():
    assert "read_file" in WORKER_PROMPT
    assert "write_file" in WORKER_PROMPT
    assert "list_files" in WORKER_PROMPT
    assert "run_command" in WORKER_PROMPT


def test_prompts_all():
    assert isinstance(__all__, list)
    assert "PLANNER_PROMPT" in __all__
    assert "TESTER_PROMPT" in __all__
    assert "WORKER_PROMPT" in __all__
    assert len(__all__) == 3
