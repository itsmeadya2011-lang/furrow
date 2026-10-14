# Furrow Tester Agent

## Role
Subagent that runs tests, linting, and type checks. Reports pass/fail status with detailed failures.

## Capabilities
- Runs test commands (pytest, npm test, cargo test, go test, etc.)
- Runs lint commands (ruff, mypy, npm lint, etc.)
- Runs type check commands (mypy, tsc, etc.)
- Parses output and determines pass/fail

## Input
- Goal: Original goal for context
- Tasks: List of completed tasks

## Output
TestResult JSON:
```json
{
  "passed": true,
  "summary": "All tests passed",
  "failures": []
}
```

LintResult JSON:
```json
{
  "passed": true,
  "summary": "All lint checks passed",
  "issues": []
}
```

## Configuration
Test, lint, and type check commands configurable via settings:
- test_commands: List of commands to try
- lint_commands: List of lint commands
- type_check_commands: List of type check commands

## Model
Uses tester_model (default: claude-3-5-sonnet-20241022)