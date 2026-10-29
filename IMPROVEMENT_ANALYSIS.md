# Furrow Codebase Improvement Analysis

This analysis identifies areas for improvement across the codebase based on code review, 
patterns, edge cases, and test coverage gaps.

## 1. Core Orchestrator (`furrow/core/orchestrator.py`)

### Issue 1.1: Inconsistent state saving behavior on early exit
**Location:** `orchestrator.py:313-325`  
**Code:**
```python
if not plan.tasks:
    self.event_bus.emit(
        Event(type=EventType.LOG, message="no tasks planned - goal complete")
    )
    self._stopped_reason = "goal complete"
    break
```
**Issue:** When plan has no tasks (goal complete), the orchestrator breaks without saving state. The README says "a cycle that ends early (a planner error, or a plan with no tasks) is not persisted, so the saved cycle count can lag the reported one by one." This is intentional per README, but could be problematic for resume - if a cycle ends with "goal complete" due to empty plan, state isn't saved. However the tests expect this behavior. This might be fine as-is if intentional.

### Issue 1.2: Potential issue with usage tracking on resume
**Location:** `orchestrator.py:97-147`  
**Code:**
```python
self._usage_seen = self._client_usage()
# Restore no-progress guard state from persisted run state
try:
    self._no_progress = int(getattr(st, "no_progress", 0) or 0)
except Exception:
    self._no_progress = 0
```
**Issue:** On resume, `_usage_seen` is set to current client usage (which starts at zero for a fresh client). But if resuming, we might want to track usage correctly across restarts. The comment says "A resumed run starts a fresh client whose counter starts at zero, so the baseline for the per-cycle delta is the client's CURRENT usage, not the persisted total." This seems correct. But if the client is shared/passed in, its usage might not be zero. Also if resuming without a fresh client instance, behavior could be odd.

### Issue 1.3: Broad exception catching
**Location:** `orchestrator.py:291-325, 326-345, 346-373`  
**Issue:** Multiple places catch `Exception as e`, log it, decrement `self.cycles -= 1` (which seems odd - maybe to not count failed cycles?), set stopped reason, break. This is defensive but the `cycles -= 1` is concerning - if a cycle fails partway through, decrementing cycles means the cycle counter could go backwards or be misleading. Also it's not clear if this is the intended behavior. Looking up how it's used - in the loop they increment at start of cycle (`self.cycles += 1` at line ~290), so on failure they decrement back. That preserves cycle count correctly. This is a bit tricky but seems intentional.

### Issue 1.4: Task merge logic is complex and has potential issues
**Location:** `orchestrator.py:453-520`  
**Issue:** The merge logic is quite complex with many branches. Line ~503-515: when old status is not completed/failed/blocked and certain conditions hold, sets to pending. The logic is a bit convoluted. Also when a task appears again in plan with same description/files and was completed, it gets marked completed and added to skip_ids - correct. But what if the plan changes other aspects? The code preserves old.description/files/dependencies and overwrites from plan? Let us look closer:
```python
old.description = pt.description
old.files = pt.files
old.dependencies = pt.dependencies
old.status = pt.status
```
So it copies from plan's processed task. But earlier it did logic to determine pt.status. Also if the task was completed and description/files changed, it sets pt.status = "pending" (line ~476-485). Correct. But there's a case where if old.status was completed and same desc/files, it sets pt.status = "completed" and adds to skip_ids. Then later copies over. That seems right.

But line ~501-515 has confusing logic - if old.status is completed/failed/blocked, it does specific handling; else (pending/running?) it does another branch. Also the else branch at line ~501 does:
```python
else:
    if old.attempts <= max_retries and (
        old.status == "failed" or old.status in ("pending", "running")
    ):
        pt.status = "pending"
    else:
        pt.status = "pending"  # Always sets to pending?
```
That seems odd - both branches set to pending. That can't be right. Let us look up the actual code again.

### Issue 1.5: Redundant/buggy logic in task merge (else branch)
**Location:** `furrow/core/orchestrator.py:390-410`  
**Code:**
```python
                elif old.status == "blocked":
                    pt.status = "blocked"
                else:
                    if old.attempts <= max_retries and (
                        old.status == "failed" or old.status in ("pending", "running")
                    ):
                        pt.status = "pending"
                    else:
                        pt.status = "pending"
                old.description = pt.description
```
**Issue:** The else branch (for statuses not completed/failed/blocked, i.e. pending/running) has an if-else that both set `pt.status = "pending"`. The condition is meaningless - it always sets pending regardless. This is dead code/redundant logic. It should probably just set `pt.status = "pending"` or have meaningful logic. Looking at how it's used - for a running task that gets re-planned, maybe we want to reset it? Or handle differently. The code is sloppy here.

### Issue 1.6: Unused variable or odd logic
**Location:** `furrow/core/orchestrator.py:503-515` in the exception handler in run_one - when worker raises exception, after setting status they do similar handling. Also notice in the first exception handler (line ~445-503) and the second one (line ~545-620) there's duplicated code for handling task failure/retry. Could be refactored, but not necessarily a bug.

### Issue 1.7: The retry logic in orchestrator
**Location:** `furrow/core/orchestrator.py:624-659`  
**Code:**
```python
        async def run_retries(tasks: list[TaskModel]) -> None:
            """Re-run tasks that failed while retries remain, until the budget is spent."""
            retry_queue = [t for t in tasks
                           if t.status == "pending" and t.attempts <= max_retries]
            while retry_queue:
                batch = retry_queue
                retry_queue = []
                await run_tasks(batch)
                for t in batch:
                    if t.status == "pending" and t.attempts <= max_retries:
                        retry_queue.append(t)
```
**Issue:** When a task fails and needs retry, `run_one` sets it to pending if `task.attempts <= max_retries`. But in our flow, `attempts` was incremented in `run_one` before deciding status. So if attempts becomes (previous_attempts + 1) and that's <= max_retries, status becomes pending. Correct. Also the retry_queue checks `t.attempts <= max_retries` - same condition. That seems right.

But also notice: when a task is requeued for retry, the same worker instance is reused (via `worker_for` which caches by task.id). The worker has its own message history built up across iterations. Reusing the worker instance across retries means it keeps the conversation history - is that intended? The worker accumulates messages as it iterates. If we retry the same task, do we want to start fresh or continue from where it left off? Looking up how WorkerAgent is used - each call to `worker.run()` does a fresh loop from iteration 0 to max_iters, building messages from scratch starting with `_initial_text()`. It doesn't preserve message history between separate `run()` calls. Let us check: in WorkerAgent.run(), it creates a new `messages` list each time starting fresh. So reusing the WorkerAgent instance is fine as long as the task object is updated - and the worker reads from `self.task`. Also the worker stores `self.files_changed` across the instance lifetime? It does `self.files_changed.add()` and returns them. So if retried, it would accumulate files_changed across retries. That might be odd/wasteful or might be intended. But the orchestrator creates a new WorkerAgent per task id on first use and reuses it. That seems intentional per the comment "Return the worker for *task*, creating it once so retries reuse it." Maybe to preserve some state.

## 2. Worker Agent (`furrow/agents/worker.py`)

### Issue 2.1: Broad exception handling in run()
**Location:** `worker.py:367-371`  
**Code:**
```python
    async def run(self) -> WorkerResult:
        try:
            return await self._run()
        except Exception as exc:  # pragma: no cover - defensive
            return WorkerResult(summary=str(exc), success=False, error=str(exc), iterations=0)
```
**Issue:** This catches all exceptions and returns a failed result. But looking up how orchestrator uses it - orchestrator also has its own try-except around `worker_for(task).run()` and handles exceptions by converting them to task failures. So if worker.run() already catches and returns a failed WorkerResult, the orchestrator won't see an exception - it will see a WorkerResult with success=False. But the orchestrator's code says "A raised exception is handled exactly like a worker that returned success=False" and has exception handling. That suggests either both paths need to work, or there's redundancy. But looking up the tests - the orchestrator test `test_worker_exception_becomes_task_failure` expects that if worker raises, the orchestrator handles it. Also looking up how the code behaves - if worker.run() catches and returns a result, then orchestrator won't get an exception, it will get the result. That would mean the orchestrator's exception handling in run_one might be redundant or for a different case. Let us look up the test again.

### Issue 2.2: Tool argument coercion is strict
**Location:** `worker.py:433-447`  
**Code:**
```python
    @staticmethod
    def _coerce_tool_args(arguments: Any) -> dict[str, Any] | None:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            text = arguments.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return None
            if isinstance(parsed, dict):
                return parsed
            return None
        return None
```
**Issue:** If arguments is a string that's valid JSON but parses to a non-dict (like a list), it returns None. The orchestrator/worker returns "error: invalid tool arguments" in that case. This is defensive but maybe could be more helpful. Also the README says "the worker's tools treat their arguments defensively as well: a tool argument value that is not an object is reported back as an `error: invalid tool arguments` tool result instead of failing the whole task." Correct behavior per tests/README.

### Issue 2.3: Command execution uses `/bin/sh -c` on POSIX
**Location:** `worker.py:515-545`  
**Issue:** On POSIX they do `asyncio.create_subprocess_exec("/bin/sh", "-c", command, cwd=str(self.workspace), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **spawn_kwargs)`. This is correct for allowing shell constructs as stated in README. But they also do `shlex.split(command)` earlier for validation? Let us look:
```python
        try:
            shlex.split(command)
        except ValueError:
            return "error: could not parse command"
```
But `shlex.split` fails on unbalanced quotes - and the command might be valid for shell in some cases? Or they use it as a basic sanity check. Also they pass the whole command string to `shlex -c` which is fine. The check is there but may reject some valid shell commands. Could be overly strict or just a sanity check. Seems okay.

### Issue 2.4: In _run_command, if os.name != "posix", they do different thing
**Location:** `worker.py:537-545`  
**Code:**
```python
            else:
                parts = shlex.split(command)
                if not parts:
                    return "error: empty command"
                proc = await asyncio.create_subprocess_exec(
                    *parts,
                    cwd=str(self.workspace),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **spawn_kwargs,
                )
```
But on non-POSIX they don't use shell -c, they split and exec directly. That means shell features like pipes, &&, redirects wouldn't work. But the worker says "run a shell command in the workspace" and README says "commands run through /bin/sh -c so ordinary shell constructs..." - this is POSIX-focused behavior. Maybe fine.

### Issue 2.5: Timeout handling in worker
**Location:** `worker.py:547-575`  
**Issue:** When timeout occurs, they call `_kill_process_tree`. On POSIX with new_session=True, they kill the process group. Correct. Also they have fallback. Seems good.

### Issue 2.6: Output truncation
**Location:** `worker.py:557-565`  
**Code:**
```python
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        combined = out + "\n[stderr]\n" + err if err.strip() else out
        cap = 20000
        if len(combined) > cap:
            combined = "[... output truncated ...]\n" + combined[-cap:]
```
They keep the tail (last cap chars). That seems reasonable for seeing recent errors.

### Issue 2.7: Hardcoded values
**Location:** `worker.py:81-152` etc.  
- `MAX_LIST_ENTRIES = 500` (line ~81) - reasonable default
- `MAX_SUBSTITUTION_DEPTH = 3` (line ~152) - reasonable for preventing infinite recursion
- Output truncation at 20000 chars in worker (line ~562) and also in read_file at 20000 (line ~492)
- These could be configurable via settings, but maybe not critical.

## 3. Tester Agent (`furrow/agents/tester.py`)

### Issue 3.1: Test detection logic complexity
**Location:** `tester.py:271-370`  
**Issue:** The test detection logic is a bit complex. For Python projects with tests/ dir, it prefers pytest even if package.json exists (line ~310-320). That makes sense. But the logic flow could be cleaner. Also it has some redundancy.

### Issue 3.2: parse_test_output might miss some cases
**Location:** `tester.py:134-269`  
**Issue:** The parsing is heuristic-based. It looks for pytest-style "FAILED file::test" patterns, file:line:col, etc. But different test runners format differently. The code tries to be defensive which is good. Also it has fallback to LLM parsing - the tester uses both the raw output parsing and LLM analysis. That seems robust.

### Issue 3.3: _looks_like_failure logic
**Location:** `tester.py:512-546`  
**Issue:** The logic to distinguish real failures from "0 failed" messages is clever with regexes. But regexes can be brittle. Also line ~545 does `return any(k in remainder.lower() for k in _FAILURE_KEYWORDS)` - but if after removing zero-count phrases nothing is left but the keyword appears in a negative context? Maybe okay. This seems well thought out based on the README saying it handles this case.

### Issue 3.4: When LLM fails to parse or is unavailable
**Location:** `tester.py:447-510`  
**Code:**
```python
            llm_result: TestResult | None = None
            if not spawn_failed:
                # A runner that never started has no output worth summarising.
                prompt = TESTER_PROMPT + "\n\nGoal: " + goal + "\n\nTest output:\n" + combined
                try:
                    response = await self.client.complete(
                        prompt, system=TESTER_SYSTEM, model=self.settings.tester_model
                    )
                    llm_result = parse_test_result(response)
                except Exception:
                    llm_result = None
```
**Issue:** If LLM call fails, they set llm_result=None and fall back to raw parsing. Also in _finalize, if llm_result is None they do fallback parsing. That seems reasonable for robustness.

## 4. LLM Client (`furrow/llm.py`)

### Issue 4.1: Error handling in retry
**Location:** `llm.py:305-359`  
**Issue:** The retry logic uses tenacity if available, otherwise manual retry. Both seem reasonable. But when tenacity is used, it retries on `_is_retryable` exceptions. For manual retry, same logic. Good.

### Issue 4.2: Anthropic/OpenAI message conversion
**Location:** `llm.py:199-303`  
**Issue:** The conversion logic looks complex but necessary for supporting different providers. Seems okay.

### Issue 4.3: Cost estimation might fail silently
**Location:** `llm.py:429-430, 458-459, 499-500, 556-557`  
**Code:**
```python
        try:
            usage.cost_usd = estimate_cost(model, usage, self.settings.pricing_overrides())
        except Exception:
            usage.cost_usd = 0.0
```
**Issue:** Cost estimation failures are silently ignored and cost set to 0.0. That seems reasonable as a fallback. Also README says "Only token counts are tracked — cost estimation in currency is not implemented yet." But there's cost estimation code - maybe partially implemented.

### Issue 4.4: No rate limiting
**Issue:** No explicit rate limiting beyond retries. Could be an issue in some cases, but maybe out of scope.

## 5. Config (`furrow/config.py`)

### Issue 5.1: Hardcoded defaults
**Location:** `config.py:180-240`  
**Issue:** Many defaults are hardcoded. Also some models are very specific (old Claude model versions). The defaults might be outdated. But configurable via env vars, so not terrible.

### Issue 5.2: max_retries vs max_task_retries
**Location:** `config.py:228-229`  
```python
    max_retries: int = Field(default=3, ge=0)
    max_task_retries: int = Field(default=1, ge=0)
```
**Issue:** Two different retry concepts - LLM request retries vs task retries. The naming is a bit confusing. Also README says "FURROW_MAX_RETRIES | 3 | Retry attempts for transient LLM errors" and "FURROW_MAX_TASK_RETRIES | 1 | Times a failed task is retried before it is marked failed". So max_retries is for LLM calls, max_task_retries is for failed tasks. The separation makes sense but could be clearer in naming.

## 6. State (`furrow/core/state.py`)

### Issue 6.1: State versioning
**Location:** `state.py:59-60, 114-150`  
**Code:**
```python
STATE_VERSION = 1
...
class RunState(BaseModel):
    version: int = STATE_VERSION
    ...
```
**Issue:** There's a version field but no migration logic if version changes. If the schema evolves, old state files won't be migrated - they'd fail validation or cause issues. But load() does `RunState.model_validate_json(raw)`. If the schema changes in incompatible ways, Pydantic validation might fail and it returns None (corrupt file case). That seems reasonable as a safe fallback.

### Issue 6.2: Atomic write is good
**Location:** `state.py:180-230`  
**Issue:** The atomic write with temp file, flush, fsync, os.replace is well implemented. Good practice.

## 7. CLI (`furrow/cli/main.py`)

### Issue 7.1: Exit codes
**Location:** `cli/main.py:285-345`  
**Issue:** The exit code logic seems correct based on README.

### Issue 7.2: _run_settings creates new Settings instance
**Location:** `cli/main.py:87-115`  
**Issue:** It does `Settings.model_validate(merged)` which validates. Good.

## 8. Web Server (`furrow/web/server.py`)

### Issue 8.1: Global state
**Location:** `web/server.py:65-90`  
**Code:**
```python
# Registry of the run that is currently active (or of the last one that ran).
_CURRENT: dict[str, Any] = dict(DEFAULT_CURRENT)

# Strong references to background tasks so they are never garbage collected
# while they are still pending.
_TASKS: set[asyncio.Task[RunResult]] = set()
```
**Issue:** Global mutable state. In a single-process server this is fine. But could cause issues in tests or multi-process scenarios. Also the code seems to handle cleanup.

### Issue 8.2: Web UI is embedded as a big string
**Location:** `web/server.py:93-380`  
**Issue:** The entire HTML/JS/CSS is embedded as a raw string. Hard to maintain, but functional.

### Issue 8.3: Potential XSS? The UI uses textContent for user data
**Location:** Looking at the JS in the HTML - they use `textContent` for most things, which is good for XSS prevention. Also they have `SAFE_STATUSES` mapping. The UI seems reasonably safe.

## 9. Tests

### Issue 9.1: Test coverage gaps
Let us look up some areas that might not be well tested.

**In orchestrator:** What about when a task is blocked and then in next cycle? The blocked task stays blocked - tested. What about if dependencies change? The merge logic updates dependencies from plan (`old.dependencies = pt.dependencies` line ~408). That seems right.

**In worker:** Command denial edge cases - the worker has extensive security logic. Tests probably cover it. Let us look up test_worker for security.

**In parsing:** Edge cases for JSON parsing from LLM responses.

**Missing tests?** For example:
- Orchestrator: what happens if state save fails mid-run? The code emits ERROR event and continues (line ~429-451 in orchestrator). Is there a test for this?
- Orchestrator: behavior when cycles reach max_cycles vs goal complete - different stopped reasons
- Worker: what if tool returns invalid args? Returns "error: invalid tool arguments" - tested?
- LLM: retry behavior
- Web: API endpoints

### Issue 9.2: Orchestrator state save failure not explicitly tested?
**Location:** `orchestrator.py:429-451` - when state save fails, it catches exception, emits ERROR event with message including path, does NOT emit STATE_SAVED. This is implemented but I don't see a test for it. That might be a gap in test coverage.

### Issue 9.3: More edge cases in worker security
The worker has very complex security logic. Let us look up if all cases are tested. Also the worker denies redirections outside workspace - tested? Let us search.

Also in worker: when command has unparseable shell syntax and shlex.split fails in validation, it returns "error: could not parse command" (line ~555-559). But then later when actually executing on POSIX it does `"/bin/sh", "-c", command` regardless. The validation uses shlex.split on the whole command - but shlex.split is more strict than shell. Also complex shell commands might fail shlex.split. That could cause false positives. That seems problematic.

**Location:** `worker.py:555-559`  
```python
        try:
            shlex.split(command)
        except ValueError:
            return "error: could not parse command"
```
**Issue:** This validation rejects commands that shlex can't parse even if they're valid shell syntax (like certain quoting). That could break legitimate commands. For example, some shell constructs might not parse cleanly with shlex.split in non-POSIX mode or with certain syntax. That seems overly strict and could break valid use cases.

Also they do this validation AFTER the security check (`is_command_denied`). The security check already does its own parsing. This extra check might be redundant or harmful.

### Issue 9.4: In worker, when running on POSIX they always use shlex with shell
But the validation rejects if shlex.split fails on the whole command. That doesn't make sense because passing to `sh -c` doesn't require the command to be shlex-splittable as a single unit - the shell parses it. For example, `echo "hello world"` - shlex.split would split it fine. But maybe some cases? Or perhaps the intent is different. This could be a bug.

Also looking up how it's used - they want to validate it's parseable. But it's a weak validation. Maybe better to just try to execute. This could reject valid commands.

## 10. Other issues

### Issue 10.1: In orchestrator, when worker returns result, they do a lot of processing
But if the worker for some reason behaves badly, they have nested try-excepts. The code is defensive.

### Issue 10.2: In LLM client, when converting messages for OpenAI, the tool result handling
**Location:** `llm.py:277-303`  
**Issue:** When role is "user" and content is a list with tool_result blocks, it converts them to role "tool" messages in OpenAI format. But it also collects text parts from text blocks in user messages and adds them as a separate user message if there are any. The order/handling might be complex. Could have issues with message ordering.

### Issue 10.3: In orchestrator, _plan_waves does cycle detection
**Location:** `orchestrator.py:452-467`  
```python
    def _plan_waves(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        ids = {t.id for t in tasks}
        waves: list[list[TaskModel]] = []
        done: set[str] = set()
        remaining = list(tasks)
        while remaining:
            wave = [t for t in remaining if all(d in done or d not in ids for d in t.dependencies)]
            if not wave:
                for t in remaining:
                    t.status = "blocked"
                break
            waves.append(wave)
            for t in wave:
                done.add(t.id)
            remaining = [t for t in remaining if t not in wave]
        return waves
```
**Issue:** If there's a cycle (A depends on B depends on A), when no wave can be formed, it marks all remaining as blocked and breaks. That prevents infinite loop. But it marks ALL remaining as blocked - even if some might not be in the cycle? For example if tasks form a cycle plus some independent tasks that got stuck? No wait - it processes waves in order of dependencies. If after processing what's possible, remaining tasks all have dependencies that form cycles or depend on each other in a cycle with none done yet in that remaining set, then it blocks them all. That seems reasonable as a safeguard.

### Issue 10.4: Missing type hints/validation
In some places, types could be more precise. Also in config, some defaults use mutable defaults? Looking up config.py - they use `Field(default_factory=...)` which is correct. Good.

### Issue 10.5: Logging
**Location:** `furrow/logging.py` exists. Let us look at it.
```python
cat /workspace/app/furrow/logging.py
```
But might be simple. Also the orchestrator/others use `structlog`? The pyproject says structlog in dependencies. Let us check.

## Summary of Key Improvements

### Critical/Bug-prone
1. **Worker command validation** (`worker.py:555-559`): Using `shlex.split()` to validate shell commands is overly strict and may reject valid shell syntax. Since commands are executed via `/bin/sh -c`, the shell parser should be the source of truth. This could break legitimate complex shell commands.
2. **Orchestrator merge logic else branch** (`orchestrator.py:400-410`): Redundant if-else where both branches do the same thing - dead code that should be cleaned up.
3. **Worker exception handling duplication**: Both worker.run() and orchestrator catch exceptions - redundant but maybe intentional for defense in depth.

### Code Quality/Refactoring
4. **Orchestrator._merge_plan complexity**: The merge logic is quite complex with many branches. Could be refactored for clarity and testability.
5. **Large embedded HTML in web server**: Makes maintenance difficult. Could be moved to separate files/templates.
6. **Duplicated exception handling in orchestrator._execute**: The exception handling code in run_one is duplicated (lines ~445-503 and ~545-620 are very similar). Should be extracted into a helper method.
7. **Test detection logic in tester** could be simplified.

### Robustness/Edge Cases
8. **State migration**: No migration logic for STATE_VERSION changes - old state files might fail validation.
9. **Tool argument coercion**: Could be more user-friendly with better error messages.
10. **No explicit test for state save failure** in orchestrator - the code handles it but might not be tested.

### Performance
11. **Worker caches workers by task.id** and reuses them - this preserves files_changed across retries which might be intentional, but could also cause issues if state isn't managed carefully.
12. **In worker._list_files, it does `client.list_files` which calls `rglob("*")` on the whole directory tree** - could be expensive for large directories. The worker caps results to MAX_LIST_ENTRIES (500) which helps. Good.

### Configuration
13. **Hardcoded model defaults** are outdated (Claude 3.5 models). Could be updated, but not necessarily a bug.
14. **Some magic numbers** (truncation limits, timeouts) could be more configurable.

### Testing Gaps
15. State save failure handling in orchestrator not obviously tested
16. Some edge cases in worker security (complex command substitutions)
17. Web API error cases

## Specific Code Improvements with Examples

### A. Remove redundant code in orchestrator merge
**File:** `furrow/core/orchestrator.py:400-410`
```python
                elif old.status == "blocked":
                    pt.status = "blocked"
                else:
                    if old.attempts <= max_retries and (
                        old.status == "failed" or old.status in ("pending", "running")
                    ):
                        pt.status = "pending"
                    else:
                        pt.status = "pending"
```
Should become:
```python
                elif old.status == "blocked":
                    pt.status = "blocked"
                else:
                    pt.status = "pending"
```
The condition is meaningless as written.

### B. Fix overly strict command validation in worker
**File:** `furrow/agents/worker.py:555-559`
```python
        try:
            shlex.split(command)
        except ValueError:
            return "error: could not parse command"
```
This check uses `shlex.split` which has different parsing rules than shell. It can reject valid shell commands. Since the command is executed via `/bin/sh -c` on POSIX (which is the main case), and the security check already does thorough parsing, this validation is likely too strict and could cause false rejections. Better to remove it or replace with a more lenient check.

### C. Extract duplicate exception handling in orchestrator
**File:** `furrow/core/orchestrator.py` - The exception handling in `run_one` is duplicated. Lines ~445-503 and ~545-620 have very similar code for handling task failures. This should be extracted into a helper method like `_handle_task_exception(task, exc, max_retries)` or similar to reduce duplication and improve maintainability.

### D. Improve error messages
**File:** `furrow/llm.py:417, 452` - When API key missing, errors could be more helpful suggesting which env var to set. The code already says "ANTHROPIC_API_KEY is not set" which is decent.

### E. Add validation for empty goals
**File:** `furrow/cli/main.py` - The CLI handles empty goals by prompting, which is good. Web API also checks. Seems okay.

### F. In worker, the files_changed tracking
When a worker retries, it reuses the same worker instance (so `self.files_changed` accumulates across retries). Also the orchestrator adds files_changed from the result to the task. But if the worker is reused and accumulates files_changed internally, and also returns them each time - could lead to duplicates or confusion. But they use sets internally and convert to sorted lists. Also the task.files_changed gets overwritten from the result each retry? Let us look: in run_one, when processing result (line ~545+ in the actual result handling), they do `task.files_changed = sorted({str(p) for p in (getattr(result, "files_changed", None) or [])})` - so they overwrite from the result each time. The worker returns `sorted(self.files_changed)` each run. Since the worker instance is reused and accumulates, but they get reset each call to run()? Let us check WorkerAgent.run() - it creates fresh messages each call but doesn't reset self.files_changed. Also in _write_file they add to self.files_changed. So if the same worker instance is used across retries and makes more writes, files_changed would accumulate. That might be wrong - we probably want files_changed for just that attempt's work, or we want all files changed across all attempts. But looking up how it's used - the task.files_changed gets overwritten each attempt with the result from that attempt. So if a retry happens, the result from the retry overwrites it. That means files_changed only reflects the last attempt. Is that intended? Maybe. Or maybe we want to union them. Hard to say, but the current behavior overwrites each time.

### G. In orchestrator, when a task fails due to exception in run_one, they do set files_changed? Looking up - in the exception handler in run_one (both places), they don't set task.files_changed. They set error, attempts, status etc. But if the worker had made file changes before failing? Unclear. But the worker might have partially modified files.

### H. Potential issue with task result/error preservation in merge
**Location:** `orchestrator.py:421-430`
```python
                # A freshly planned task carries result=None and error=None,
                # so never let that wipe a result or error already recorded by
                # a previous cycle. Only adopt non-None values from the plan.
                if pt.result is not None:
                    old.result = pt.result
                if pt.error is not None:
                    old.error = pt.error
```
**Issue:** They only copy result/error from plan if they're not None. But what if the plan legitimately wants to clear them? Probably rare. Also they copy description/files/dependencies from plan always. That seems right per the tests.

## Recommendations

1. **Refactor orchestrator**: Extract duplicate exception handling in `_execute().run_one()`, simplify the merge logic (remove redundant else branch), and consider breaking down the complex methods.

2. **Fix worker command validation**: Remove or make more lenient the `shlex.split()` check on line ~555-559 in worker.py, as it can reject valid shell commands. The security checks in `is_command_denied()` are more thorough anyway.

3. **Improve error messages**: Add more context to errors where helpful, especially around configuration issues.

4. **Add tests** for:
   - State save failure in orchestrator (error event emitted, run continues)
   - More edge cases in worker command parsing/validation
   - Behavior when state file is corrupted vs missing

5. **Consider moving web UI**: The embedded HTML/CSS/JS makes the file very long and hard to maintain. Could load from static files.

6. **Clean up TODOs/FIXMEs**: None found currently, which is good.

7. **Type safety**: Some places could use more specific types. For example in orchestrator, `self._console_obj` uses a sentinel - could use `Optional[Console]` with None as default differently.

8. **Documentation**: The codebase is well-documented with comments. Good.

## Conclusion

The codebase is generally well-structured with good defensive programming practices. The main areas for improvement are:
- Removing redundant/dead code (like the else branch in merge logic)
- Fixing potentially overly strict validation in the worker
- Reducing code duplication in the orchestrator
- Possibly refactoring large complex methods
- Adding tests for some edge cases

Most of these are code quality improvements rather than critical bugs. The codebase appears to be in good shape overall.
