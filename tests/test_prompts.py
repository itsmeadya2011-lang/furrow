from furrow.agents.prompts import PLANNER_PROMPT, TESTER_PROMPT, WORKER_PROMPT


def test_prompts_exist():
    assert isinstance(PLANNER_PROMPT, str)
    assert isinstance(WORKER_PROMPT, str)
    assert isinstance(TESTER_PROMPT, str)
    assert len(PLANNER_PROMPT) > 0
    assert len(WORKER_PROMPT) > 0
    assert len(TESTER_PROMPT) > 0


def test_planner_prompt_contains_json_shape():
    assert '"tasks"' in PLANNER_PROMPT
    assert '"id"' in PLANNER_PROMPT
    assert '"description"' in PLANNER_PROMPT
    assert '"rationale"' in PLANNER_PROMPT


def test_worker_prompt_contains_tools():
    assert "read_file" in WORKER_PROMPT
    assert "write_file" in WORKER_PROMPT
    assert "list_files" in WORKER_PROMPT
    assert "run_command" in WORKER_PROMPT


def test_tester_prompt_contains_json_shape():
    assert '"passed"' in TESTER_PROMPT
    assert '"summary"' in TESTER_PROMPT
    assert '"failures"' in TESTER_PROMPT


def test_prompts_are_module_level():
    """Ensure prompts are defined at module level (importable)."""
    import furrow.agents.prompts as prompts_module

    assert hasattr(prompts_module, "PLANNER_PROMPT")
    assert hasattr(prompts_module, "WORKER_PROMPT")
    assert hasattr(prompts_module, "TESTER_PROMPT")