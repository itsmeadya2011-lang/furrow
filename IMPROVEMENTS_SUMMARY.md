# Furrow Improvements Summary

This document summarizes the improvements made to the Furrow autonomous coding agent.

## 1. Code Refactoring & Maintainability

### Worker Agent Refactoring
- **Split `worker.py` (821 lines) into modular package:**
  - `furrow/agents/worker/__init__.py` - Main exports
  - `furrow/agents/worker/agent.py` - WorkerAgent class
  - `furrow/agents/worker/tools.py` - Tool implementations (read_file, write_file, list_files, run_command, git)
  - `furrow/agents/worker/command_policy.py` - Command validation and security policy

### Benefits
- Better separation of concerns
- Easier testing and maintenance
- Clearer code organization
- Backward compatibility maintained

## 2. New Code Review Agent

### `furrow/agents/code_review.py`
- **CodeReviewAgent** - Additional verification layer that reviews changes before tests run
- Checks for: correctness, security, style, architecture, testing, performance
- Integrates into orchestrator loop between execution and testing
- Uses new `reviewer_model` setting (configurable via `FURROW_REVIEWER_MODEL`)
- Emits `CODE_REVIEW_STARTED` and `CODE_REVIEW_FINISHED` events
- If review fails, adds issues to feedback and continues to next cycle (skips tests)

### Configuration
- Added `CodeReviewResult` model to config
- Added `reviewer_model` setting to Settings

## 3. Git Integration

### `furrow/agents/git_tools.py`
- **Git tools for worker agent:**
  - `git_status` - Show working tree status
  - `git_diff` - Show changes (staged/unstaged)
  - `git_log` - Show commit history
  - `git_branch` - Show current branch
  - `git_commit` - Commit changes with message
  - `git_checkout` - Switch/create branches

- Added to `WORKER_TOOLS` in prompts.py
- Updated `WORKER_SYSTEM` prompt to mention git tools

## 4. Improved Test Runner

### Enhanced `TesterAgent` in `furrow/agents/tester.py`
- **Parallel test execution** - Support for `pytest-xdist` via `-n auto` flag
- **Test file filtering** - Include patterns (`test_patterns`) and exclude patterns (`test_exclude_patterns`)
- **New Settings:**
  - `test_parallel` - Enable parallel test execution
  - `test_patterns` - List of patterns to include (pytest `-k`)
  - `test_exclude_patterns` - List of patterns to exclude (pytest `--deselect`)

### CLI Options
- `--test-parallel/--no-test-parallel`
- `--test-pattern` (multiple)
- `--test-exclude` (multiple)

### Web API
- Added to `StartRequest` and `SocketRequest` models

## 5. Custom Tool Support

### `furrow/agents/custom_tools.py`
- **ToolRegistry** - Registry for custom tools
- **CustomTool** - Dataclass for tool definition
- **Decorator API** - `@create_tool()` for easy tool creation
- **Global registry** - `register_tool()`, `unregister_tool()`, `get_custom_tools()`
- **Integration** - Custom tools automatically added to worker's available tools

### Usage Example
```python
from furrow.agents import create_tool

@create_tool(
    "my_tool",
    "Description of my tool",
    {"type": "object", "properties": {"arg1": {"type": "string"}}, "required": ["arg1"]},
)
async def my_tool(workspace: Path, args: dict[str, Any]) -> str:
    return f"Result: {args['arg1']}"
```

## 6. Improved Error Handling

### `furrow/errors.py`
- **Structured error hierarchy:**
  - `FurrowError` (base)
  - `ConfigurationError`
  - `ProviderError`
  - `ToolError`
  - `SecurityError`
  - `StateError`
  - `TimeoutError`
  - `ValidationError`

- **Utilities:**
  - `format_error()` - Format exception for logging/serialization
  - `handle_error()` - Handle error with logging

### Orchestrator Integration
- Added `_handle_error()` method for consistent error handling
- All try/except blocks now use structured error handling
- Errors include error codes and details for better debugging

## 7. Configuration Enhancements

### New Settings in `furrow/config.py`
- `reviewer_model` - Model for code review agent
- `test_parallel` - Enable parallel test execution
- `test_patterns` - Test file patterns to include
- `test_exclude_patterns` - Test file patterns to exclude

### New Event Types in `furrow/core/events.py`
- `CODE_REVIEW_STARTED`
- `CODE_REVIEW_FINISHED`

## 8. Web UI Updates

### `furrow/web/server.py`
- Added handlers for code review events in JavaScript UI
- Updated `StartRequest` and `SocketRequest` models with new test options
- Updated `_build_orchestrator` and `_run_settings` to pass new settings

## 9. CLI Updates

### `furrow/cli/main.py`
- Updated `_run_settings()` to accept new parameters
- Added CLI options for test configuration to `start` and `resume` commands

## File Structure Changes

```
furrow/
├── agents/
│   ├── __init__.py              # Updated exports
│   ├── planner.py
│   ├── tester.py                # Enhanced with test filtering/parallel
│   ├── worker.py                # Refactored (thin wrapper)
│   ├── worker/                  # NEW: Worker package
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── tools.py
│   │   └── command_policy.py
│   ├── code_review.py           # NEW: Code review agent
│   ├── git_tools.py             # NEW: Git integration
│   ├── custom_tools.py          # NEW: Custom tool support
│   ├── prompts.py               # Updated with git tools & dynamic tools
│   └── parsing.py
├── core/
│   ├── orchestrator.py          # Updated with code review & error handling
│   ├── events.py                # Added code review event types
│   ├── state.py
│   └── events.py
├── errors.py                    # NEW: Structured error handling
├── config.py                    # Added new settings
├── cli/
│   └── main.py                  # Updated with new CLI options
├── web/
│   └── server.py                # Updated with new settings & events
├── logging.py
├── llm.py
├── costs.py
└── __init__.py                  # Added error exports
```

## Backward Compatibility

All changes maintain backward compatibility:
- Original `worker.py` imports still work
- Existing CLI commands work unchanged
- Existing web API endpoints work unchanged
- Settings have sensible defaults
- No breaking changes to public APIs

## Testing

The existing test suite should continue to work. New features can be tested by:
1. Running with `--test-parallel` flag
2. Using `--test-pattern` and `--test-exclude`
3. Registering custom tools via `@create_tool`
4. Using git tools in worker tasks
5. Observing code review events in web UI

## Future Enhancements (Not Implemented)

- More sophisticated test runner (Jest, Vitest, Cargo, Go test parallel support)
- Code review agent with more detailed analysis
- Git integration with remote operations (push, pull, fetch)
- Custom tool marketplace/sharing
- More error types and recovery strategies