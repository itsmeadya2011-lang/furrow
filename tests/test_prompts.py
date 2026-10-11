from furrow.agents.prompts import (
    PLANNER_PROMPT,
    TESTER_PROMPT,
    WORKER_PROMPT,
)


def test_planner_prompt_has_required_content():
    assert "tasks" in PLANNER_PROMPT
    assert "rationale" in PLANNER_PROMPT


def test_worker_prompt_has_required_content():
    assert "SUMMARY" in WORKER_PROMPT
    assert "FILE:" in WORKER_PROMPT


def test_tester_prompt_has_required_content():
    assert "passed" in TESTER_PROMPT.lower()
    assert "failures" in TESTER_PROMPT.lower()


def test_planner_prompt_has_json_shape():
    assert "id" in PLANNER_PROMPT
    assert "description" in PLANNER_PROMPT
    assert "dependencies" in PLANNER_PROMPT


def test_worker_prompt_has_files_changed():
    assert "FILES_CHANGED" in WORKER_PROMPT
    assert "CONTENT:" in WORKER_PROMPT
    assert "ENDFILE" in WORKER_PROMPT


def test_tester_prompt_has_json_shape():
    assert "passed" in TESTER_PROMPT
    assert "summary" in TESTER_PROMPT
    assert "failures" in TESTER_PROMPT