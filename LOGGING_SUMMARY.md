Summary of changes:

1. Created `/workspace/app/furrow/logging.py`:
   - Added `setup_logging(log_level: str = "INFO")` function that configures structlog with console renderer (timestamps, log level).
   - Added `get_logger(name: str)` function that returns a named structlog logger.

2. Updated `/workspace/app/furrow/core/orchestrator.py`:
   - Added `logger = get_logger(__name__)` after imports.
   - Added `logger.info("Starting Furrow orchestration", goal=self.goal)` at start of `run`.
   - Added `logger.info("Starting cycle", cycle=self.cycles)` at start of each cycle.
   - Added `logger.info("Goal completed", cycle=self.cycles)` when goal is met.
   - Added `logger.info("Planning started")` and `logger.info("Planning completed")` in `_cycle`.
   - Added `logger.info("Testing started")` and `logger.info("Testing completed")` in `_cycle`.

3. Updated `/workspace/app/furrow/agents/planner.py`:
   - Already had `logger = get_logger(f"{__name__}.PlannerAgent")` and logging calls for planning start/completion.

4. Updated `/workspace/app/furrow/agents/worker.py`:
   - Already had `logger = get_logger(f"{__name__}.WorkerAgent")` and logging calls for task execution start/completion.

5. Updated `/workspace/app/furrow/agents/tester.py`:
   - Already had `logger = get_logger(f"{__name__}.TesterAgent")` and logging calls for test run start, output, and completion.

6. Updated `/workspace/app/furrow/llm.py`:
   - Already had `logger = get_logger(__name__)` and debug logging for API calls (provider, model, prompt preview) and file operations.

All changes maintain existing rich console output for user-facing progress; structlog complements it.