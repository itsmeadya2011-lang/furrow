Summary of changes:

1. Created `/workspace/app/furrow/logging.py`:
   - Added `setup_logging(log_level: str = "INFO")` function that configures structlog with console renderer (timestamps, log level).
   - Added `get_logger(name: str)` function that returns a named structlog logger.

2. Updated `/workspace/app/furrow/core/orchestrator.py`:
   - Added `logger = get_logger(__name__)` after imports.
   - Added `logger.info("cycle_start", cycle=self.cycles)` after the cycle print.
   - Added `logger.info("cycle_done", cycle=self.cycles)` at end of _cycle.

3. Updated `/workspace/app/furrow/agents/planner.py`:
   - Changed to use module-level logger: `logger = get_logger(__name__)`.
   - Removed instance logger assignment in __init__.
   - Added `logger.info("planning", goal=goal)` at start of plan() method.
   - Changed existing logging calls to use module-level logger.

4. Updated `/workspace/app/furrow/agents/worker.py`:
   - Changed to use module-level logger: `logger = get_logger(__name__)`.
   - Removed instance logger assignment in __init__.
   - Added `logger.info("worker_start", task_id=self.task.id)` at start of run() · method.
   - Changed existing logging calls to use module-level logger.

5. Updated `/workspace/app/furrow/agents/tester.py`:
   - Changed to use module-level logger: `logger = get_logger(__name__)`.
   - Removed instance logger assignment in __init__.
   - Added `logger.info("test_result", passed=test_result.passed)` after getting test result in both try and except blocks.
   - Changed existing logging calls to use module-level logger.

All changes maintain existing rich console output for user-facing progress; structlog complements it.