import pytest

from furrow.agents import prompts
from furrow.agents.prompts import (
    PLANNER_PROMPT,
    PLANNER_SYSTEM,
    TESTER_PROMPT,
    TESTER_SYSTEM,
    WORKER_SYSTEM,
    WORKER_TOOLS,
)

PROMPT_CONSTANTS = [
    ("PLANNER_PROMPT", PLANNER_PROMPT),
    ("TESTER_PROMPT", TESTER_PROMPT),
    ("PLANNER_SYSTEM", PLANNER_SYSTEM),
    ("WORKER_SYSTEM", WORKER_SYSTEM),
    ("TESTER_SYSTEM", TESTER_SYSTEM),
]


@pytest.mark.parametrize(("name", "value"), PROMPT_CONSTANTS)
def test_prompt_constants_are_non_empty_strings(name: str, value: object) -> None:
    assert isinstance(value, str), f"{name} should be a str"
    assert value.strip(), f"{name} should be non-empty"


def test_worker_tools_is_list_of_four_tools() -> None:
    assert isinstance(WORKER_TOOLS, list)
    assert len(WORKER_TOOLS) == 4


def test_worker_tools_names() -> None:
    names = [tool["name"] for tool in WORKER_TOOLS]
    assert names == ["read_file", "write_file", "list_files", "run_command"]


@pytest.mark.parametrize("tool", WORKER_TOOLS, ids=[t["name"] for t in WORKER_TOOLS])
def test_worker_tool_has_required_schema_keys(tool: dict) -> None:
    assert set(tool.keys()) >= {"name", "description", "parameters"}
    params = tool["parameters"]
    assert params["type"] == "object"
    assert "properties" in params
    assert "required" in params
    assert isinstance(params["properties"], dict)
    assert isinstance(params["required"], list)


def _tool_by_name(name: str) -> dict:
    for tool in WORKER_TOOLS:
        if tool["name"] == name:
            return tool
    raise AssertionError(f"tool not found: {name}")


def test_read_file_schema() -> None:
    tool = _tool_by_name("read_file")
    assert tool["parameters"]["required"] == ["path"]
    assert "path" in tool["parameters"]["properties"]


def test_write_file_schema() -> None:
    tool = _tool_by_name("write_file")
    assert tool["parameters"]["required"] == ["path", "content"]
    props = tool["parameters"]["properties"]
    assert "path" in props
    assert "content" in props


def test_list_files_schema() -> None:
    tool = _tool_by_name("list_files")
    assert tool["parameters"]["required"] == []
    assert "path" in tool["parameters"]["properties"]


def test_run_command_schema() -> None:
    tool = _tool_by_name("run_command")
    assert tool["parameters"]["required"] == ["command"]
    assert "command" in tool["parameters"]["properties"]


def test_planner_prompt_content() -> None:
    for phrase in ["JSON", "tasks", "id", "description", "files", "dependencies", "rationale"]:
        assert phrase in PLANNER_PROMPT, f"PLANNER_PROMPT missing {phrase!r}"


def test_planner_system_content() -> None:
    lower = PLANNER_SYSTEM.lower()
    assert "planning agent" in lower
    assert "furrow" in lower
    assert "json" in lower


def test_tester_prompt_content() -> None:
    for phrase in ["passed", "summary", "failures", "JSON"]:
        assert phrase in TESTER_PROMPT, f"TESTER_PROMPT missing {phrase!r}"


def test_tester_system_content() -> None:
    lower = TESTER_SYSTEM.lower()
    assert "verification agent" in lower
    assert "furrow" in lower
    assert "json" in lower


def test_worker_system_content() -> None:
    lower = WORKER_SYSTEM.lower()
    assert "worker agent" in lower
    assert "furrow" in lower
    # mentions tools (read/write/run)
    assert "tool" in lower
    assert "read" in lower
    assert "writ" in lower  # covers write/writing
    assert "run" in lower


def test_worker_tool_descriptions_match_names() -> None:
    read_desc = _tool_by_name("read_file")["description"].lower()
    assert "read" in read_desc

    write_desc = _tool_by_name("write_file")["description"].lower()
    assert "write" in write_desc or "overwrite" in write_desc

    list_desc = _tool_by_name("list_files")["description"].lower()
    assert "list" in list_desc

    run_desc = _tool_by_name("run_command")["description"].lower()
    assert "run" in run_desc or "shell" in run_desc


def test_planner_prompt_json_shape_matches_task_fields() -> None:
    # Verify the JSON shape described in planner prompt includes TaskModel-like fields
    for field in ["id", "description", "files", "dependencies", "rationale"]:
        assert field in PLANNER_PROMPT, f"PLANNER_PROMPT missing shape field {field!r}"
    # also ensure it shows the task container shape
    assert '"tasks"' in PLANNER_PROMPT


def test_module_exports_expected_constants() -> None:
    # sanity: constants accessible via module namespace too
    for name in [
        "PLANNER_PROMPT",
        "TESTER_PROMPT",
        "PLANNER_SYSTEM",
        "WORKER_SYSTEM",
        "TESTER_SYSTEM",
        "WORKER_TOOLS",
    ]:
        assert hasattr(prompts, name)
