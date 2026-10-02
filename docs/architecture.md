# Architecture

This document is for contributors. `README.md` covers installation and usage; this file
explains how the pieces fit together and where the current edges are.

## This repository contains two related things

The single most confusing thing about this repo is that "Furrow" names two independent
implementations of the same idea. They share a name and a loop shape, but they share no
code.

### 1. Kilo agent definitions (`.kilo/`)

These are prompt-only definitions consumed by the Kilo agent runtime. They drive an
interactive session, and they use Kilo's own tool-calling agents rather than direct SDK
calls.

| File | Kind | Role |
| --- | --- | --- |
| `.kilo/agent/furrow.md` | primary agent | Runs the loop: receive goal, plan, delegate, test, report, repeat |
| `.kilo/agent/furrow-planner.md` | subagent | Breaks a goal into 1-5 independent tasks, returns JSON |
| `.kilo/agent/furrow-worker.md` | subagent | Executes one assigned task with real edit tools |
| `.kilo/agent/furrow-tester.md` | subagent | Runs tests, lint and type checks, returns pass/fail JSON |
| `.kilo/command/furrow.md` | slash command | `/furrow` — invokes the `furrow` agent with `$ARGUMENTS` |

The orchestrator definition instructs Kilo to spawn real `furrow-worker` subagents via the
Task tool, and it declares `steps: 50` with `mode: primary`. `.kilo/FURROW.md` is the
human-readable summary of the same design, and `kilo.json` sets `default_agent` to `furrow`
with `bash`, `edit` and `read` all set to `allow`.

### 2. The Python package (`furrow/`)

A standalone, dependency-installable package that reimplements the same loop as a CLI and a
web service. It talks to Anthropic, OpenAI or Ollama through their own SDKs, runs the test
suite in a subprocess, and exposes the loop over HTTP and WebSocket. Nothing in `furrow/`
reads `.kilo/`, and nothing in `.kilo/` reads `furrow/`.

Consequences worth internalising:

- Behaviour can differ between the two. The Kilo workers really do edit files, because they
  have edit tools; the Python workers do not (see
  [Known limitations](#design-notes-and-known-limitations)).
- The Kilo version runs until the user stops it. The Python version has real halt
  conditions and a cycle budget.
- The `/furrow` slash command and `furrow start` are different entry points to different
  code, even though both are called "Furrow".

## The Python package

### Data models

All three live in `furrow/config.py` and are pydantic models:

```python
class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: str | None = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
```

`TaskModel.status` is mutated in place by the orchestrator after each cycle, moving between
`pending`, `completed` and `failed`. `dependencies` is carried through the planner contract
but the orchestrator does not currently schedule against it; worker parallelism is flat.

`Settings` and the `Provider` enum are also in this module. See the README configuration
table.

### `LLMClient`

`furrow/llm.py`. One class, three lazily constructed SDK clients, selected by
`settings.provider`:

```python
async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
    model = model or self.settings.model
    if self.settings.provider == Provider.ANTHROPIC:
        return await self._complete_anthropic(prompt, system, model)
    if self.settings.provider == Provider.OPENAI:
        return await self._complete_openai(prompt, system, model)
    if self.settings.provider == Provider.OLLAMA:
        return await self._complete_ollama(prompt, system, model)
    raise ValueError(f"Unsupported provider: {self.settings.provider}")
```

Each `_complete_*` method is decorated with `@retry_transient`. The API keys are read from
`settings` first and from the bare `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` environment
variables second. `complete_json()` is a thin wrapper that parses the response with
`extract_json`.

`LLMClient` also carries three small filesystem helpers — `read_file`, `write_file` (both
`aiofiles`-backed, `write_file` creating parent directories) and `list_files`. They exist
for future use; no agent calls them today.

### The three agents

`furrow/agents/` holds one class per role. Each is a thin wrapper: build a prompt from
`prompts.py`, one `LLMClient.complete` call, parse the result.

- **`PlannerAgent`** (`planner.py`) calls with `planner_model` and normalises the parsed
  payload through `build_plan`, which coerces a lone task object into a list, drops entries
  with a missing or duplicated `id`, truncates to `max_parallel_tasks`, and appends a note
  about what it dropped to `rationale`. Truncating here as well as in the worker pool keeps
  the emitted plan honest about what will actually run.
- **`WorkerAgent`** (`worker.py`) calls with `worker_model` and returns the model's text
  verbatim as the task's `result`.
- **`TesterAgent`** (`tester.py`) runs the test suite, then calls with `tester_model` to
  turn raw output into a `TestResult`.

### `Orchestrator`

`furrow/core/orchestrator.py` is the state machine. Construction takes a `goal`, an optional
`LLMClient`, an optional `on_event` callback and optional `Settings`.

The loop is `while True` over `self._cycle()`. Each cycle increments `self.cycles`, emits
`cycle_start`, and runs plan → execute → test. `CycleRecord` is the immutable per-cycle
log:

```python
@dataclass
class CycleRecord:
    """Immutable record of one completed cycle of the development loop."""

    cycle: int
    plan: Plan
    test_result: TestResult
    tasks: list[TaskModel]
```

`status()` returns the JSON-serializable snapshot used by the REST API and the web console:

```python
{
    "goal": self.goal,
    "directive": self.directive,
    "cycles": self.cycles,
    "running": self._running,
    "done": self._is_done(),
    "tasks": tasks,
    "last_test": last_test,
}
```

`tasks` describes the last recorded cycle, each entry being `id`, `description`, `status`,
`result`. `last_test` is `null` before the first cycle completes.

### Event contract

Every observable step goes through `_emit(type, message, data)`, which builds the envelope

```python
{
    "type": type,
    "cycle": self.cycles,
    "message": message,
    "data": dict(data or {}),
}
```

and calls `on_event`. The callback is wrapped in a bare `except Exception: pass` on purpose:
a broken observer must never take the development loop down.

The eight event types are `cycle_start`, `plan`, `task_start`, `task_end`, `test_result`,
`cycle_end`, `run_complete` and `run_halt`. The web layer adds three control-only frames —
`run_started`, `run_attached`, `error` — that never originate here. See the README for the
per-event payloads.

`_halt()` and `_finish()` are the two terminal paths. `run()` also handles
`asyncio.CancelledError` and generic `Exception`, emitting `run_halt` in both cases before
re-raising.

### Goal versus directive

`goal` is set once in `__init__` and never reassigned. `directive` is what the next cycle
plans against:

- after a passing test run, `directive = goal`
- after a failing test run, `directive = "Fix failing tests:\n" + "\n".join(failures)`

The tester is always called as `TesterAgent(client=self.client).run(self.goal,
plan.tasks)` — always against the original goal. The comment in the source is explicit:
`# Always test against the original goal, never a mutated one.`

### Halt conditions

Implemented as three small predicates, composed by `_is_done()`:

- `_limit_reached()` — `max_cycles > 0 and self.cycles >= max_cycles`.
- `_last_cycle_complete()` — history is non-empty, the last cycle's task list is
  non-empty, and every task status is `completed`. An empty plan explicitly returns `False`
  with the comment "An empty plan is not evidence of completion."
- `_is_done()` — `_limit_reached() or _last_cycle_complete()`.

Plus the empty-plan case: `_cycle()` returns `None` when `plan.tasks` is empty, and `run()`
halts with "Planner produced no tasks. Halting to avoid an unbounded loop."

The important subtlety, spelled out in the README: `completed` means the worker call
returned without raising. The test result is not part of the completion test, so a run can
emit `run_complete` after a cycle whose `test_result` had `passed: false`.

## Logging

Logging lives in `furrow/logging_config.py`, which exposes three functions:
`resolve_level`, `configure_logging` and `get_logger`. It is built on `structlog`, which is a
declared runtime dependency.

### Why stderr, always

`configure_logging` sends everything to **stderr**:

```python
logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric, force=True)
```

The user-facing progress output in the orchestrator and the CLI is `rich` console output on
**stdout** — panels, status spinners, colour. A log line on stdout would corrupt the
terminal rendering and would also break `furrow start "..." | tee`, because the structured
lines would interleave with the console frames. Keeping the two streams apart is the whole
point of the module.

It routes through the stdlib `logging` machinery underneath `structlog` so that one
configuration covers both `structlog` loggers and the plain `logging.getLogger` loggers other
modules obtain. `force=True` is used because an entry point owns the process: handlers
installed by an importer would otherwise capture output away from stderr. The processor chain
merges contextvars, adds the log level, timestamps in ISO-8601 UTC, renders stack info and
exception info, and finally emits `JSONRenderer` output.

### Where the level comes from

`configure_logging` takes the level as an argument; the CLI passes
`default_settings.log_level`, which is the `Settings` field read from `FURROW_LOG_LEVEL` and
defaulting to `"INFO"`. `resolve_level` accepts the standard names (`DEBUG`, `INFO`,
`WARNING`/`WARN`, `ERROR`, `CRITICAL`/`FATAL`), `NOTSET`, and bare numbers, and returns the
matching stdlib level.

An unrecognised value is **not** an error. A typo in `FURROW_LOG_LEVEL` must not stop the
agent from running, so `resolve_level` falls back to `DEFAULT_LEVEL = "INFO"`. It also rejects
`bool` explicitly, since `bool` is an `int` subclass and `True` would otherwise be read as
level `1`.

`structlog` filters by wrapping the logger in
`structlog.make_filtering_bound_logger(numeric)`, and
`_CACHE_LOGGERS_ON_FIRST_USE = False` ensures loggers handed out before configuration runs
follow the later configuration instead of being frozen at structlog's default.

### Configured by the CLI, not at import time

`configure_logging(default_settings.log_level)` is called once at the top of both the `start`
and `web` commands in `furrow/cli/main.py`. Importing `furrow` — or `furrow.logging_config`
itself — does **not** configure anything. Configuring logging is a side effect on the *host*
process, so a library import must not do it; that would mean any application embedding
Furrow found its own logging configuration silently replaced. `get_logger` is safe to call
before configuration: structlog's default configuration is harmless, it just is not the one
we want.

### Model overrides

`furrow/config.py` owns the per-run model override:

```python
MODEL_FIELDS = ("model", "planner_model", "worker_model", "tester_model")


def settings_for_model(model: str | None) -> Settings | None:
    if not model:
        return None
    return settings.model_copy(update=dict.fromkeys(MODEL_FIELDS, model))
```

`None` means "no override", so the caller keeps using the module-level singleton and the
common case is untouched. Otherwise a **copy** is returned; the global `settings` object is
never mutated. It is shared process-wide, so mutating it would let one browser tab
re-point every other concurrent run — and the CLI — at its model, permanently, with no way
to undo it.

All four fields are rewritten, not just `model`. The three agents each pass an explicit
`planner_model` / `worker_model` / `tester_model` to `LLMClient.complete`, so overriding only
the generic `model` fallback would change nothing at all: no agent ever reads it. Rewriting
the whole set is what makes the override actually reach the calls.

The override deliberately beats `FURROW_PLANNER_MODEL` and friends. The decision rule is
"more specific, more recent instruction wins": those variables are ambient configuration
chosen when the environment was set up, whereas the flag — or the web console's model box —
was named by the user just now for this run. A user who wants per-role models from the
environment simply omits the override.

Both entry points use it. `furrow/cli/main.py` imports `settings_for_model` directly;
`furrow/web/server.py` keeps a thin private `_settings_for` wrapper that delegates to it, so
the web path picks up any future change to the field list for free.

## Bounded parallelism

`max_parallel_tasks` (default `5`) bounds concurrency in two independent places.

The planner truncates the plan to at most `max_parallel_tasks` tasks, deferring the rest
with a note appended to the rationale. Then `Orchestrator._execute` enforces the cap again
at execution time, so a plan that arrives from anywhere else is still bounded:

```python
limit = self.settings.max_parallel_tasks
if limit is None or limit < 1:
    limit = 1
limit = min(limit, len(tasks))
```

The pool is an `asyncio.Queue` of `(index, task)` pairs filled up front, drained by exactly
`limit` concurrent `runner()` coroutines gathered with `asyncio.gather`. Each runner calls
`queue.get_nowait()` and returns when the queue is empty.

Why this shape:

- **Results stay positional.** Each item carries its original index and writes to
  `results[index]`, so `zip(plan.tasks, results)` in `_cycle()` correlates outcomes with
  tasks regardless of completion order.
- **No semaphore bookkeeping.** A queue of work plus N consumers is the same bound as a
  semaphore, with less bookkeeping and no risk of a `Semaphore` leak on cancellation.
- **Failures are contained.** A worker exception is captured into the result slot and the
  runner continues. One bad task marks that task `failed` and never aborts the cycle.
- **Two layers of defence.** Truncating in the planner keeps the UI honest about what will
  run; enforcing in `_execute` keeps the bound true for any input.

Worker error handling deliberately lets `asyncio.CancelledError` propagate while converting
every other exception into a failed task, with the source comment "surfaced per task, never
aborts the cycle".

## Test running

`TesterAgent._run_tests` tries candidates in a fixed order and takes the first one that
actually executes:

```python
candidates = [
    ["pytest", "-q"],
    ["python", "-m", "pytest", "-q"],
    ["npm", "test", "--", "--silent"],
    ["pnpm", "test", "--", "--silent"],
    ["yarn", "test", "--silent"],
    ["cargo", "test", "-q"],
    ["go", "test", "./..."],
]
```

`FileNotFoundError` or `NotADirectoryError` from `asyncio.create_subprocess_exec` means the
runner is not installed, so the loop continues to the next candidate. If every candidate is
missing, `_run_tests` returns the `MISSING_RUNNER` string, which explicitly states that
tests were not run and must not be reported as a pass. A missing `workspace` directory raises
`FileNotFoundError` instead, because reporting that as "no runner found" would be a lie; it
surfaces to `run()`, which converts any exception into a failed `TestResult`.

Each run is formatted with a header containing the command, working directory, exit code and
an explicit `RESULT: PASSED` or `RESULT: FAILED` line before the raw output. That marker is
the strongest signal available when the tester LLM's JSON cannot be parsed.

### Why the timeout kills the process group

The subprocess is started with `start_new_session=True`, giving it its own process group.
This matters because test runners spawn children — `npm test` forks, `pytest-xdist` forks
workers — and killing only the direct child leaves those grandchildren alive, still holding
the terminal and the workspace.

On `asyncio.TimeoutError`, `_kill_process_group` sends `SIGKILL` to the process group via
`os.killpg`, falling back to `proc.kill()` where `getpgid` / `killpg` are unavailable. It
then waits up to 5 further seconds for the process to reap. The returned text reports
`RESULT: FAILED` with an explicit "Timed out after Ns and the process group was killed", so
a hang is reported as a failure with a known cause rather than being mistaken for a missing
runner.

### Unparseable tester output

If the tester LLM's response is not valid JSON, `_text_indicates_pass` falls back to
conservative text analysis: an explicit `RESULT:` marker wins, then zero-problem phrases
(`0 failed`, `0 errors`), then clean-success phrases, then any failure token, then a weak
pass hint. The bias is deliberate — as the docstring says, "a false pass makes the
orchestrator stop asking for fixes".

## Structured output

`extract_json` in `furrow/core/json_utils.py` is the only way model output is parsed. Bare
`json.loads` on a model response is unreliable in practice because models wrap JSON in prose
("Here's the plan you asked for:") and in markdown fences (` ```json `), both of which are
strongly discouraged by the prompts but not guaranteed absent.

`extract_json` handles three shapes in order:

1. **Bare JSON.** `json.loads(source)` first, so a valid whole-string document always wins
   over a scan.
2. **A single fenced block.** `_strip_code_fence` accepts an opening ```` ``` ```` with an
   info string in `{"", "json", "json5", "jsonc"}` and a matching closing fence, and
   returns the body. A fence with any other language tag is not treated as JSON.
3. **JSON embedded in prose.** `_scan_balanced` walks the text looking for the first `{` or
   `[`, then tracks depth while respecting string literals and backslash escapes, so a brace
   inside a quoted value does not end the scan early. It returns the first balanced block.

If nothing parses, it raises `ValueError` including up to 200 characters of the offending
text, which is what makes the planner's and tester's "Response: …" error messages useful.

The planner and tester both route their parse failures through
`extract_json` plus their own normalisation (`build_plan`, `TestResult(**data)`), and both
tolerate fences and surrounding commentary by design.

## Web run retention and event replay

`furrow/web/server.py` holds four process-global registries: `_RUNS`,
`_RUN_TASKS`, `_STREAMS` and `_FINISHED_RUNS`. Two bounded mechanisms keep them from growing
without limit.

### Retention: the 50 most recently finished runs

`_FINISHED_RETENTION = 50`. The `_FINISHED_RUNS` deque accumulates run ids in completion
order, appended only by the run task's own done callback, so a run that is still executing
can never be a candidate for eviction. When the deque overflows, `_prune_finished_runs`
evicts from the oldest end, calling `_evict` to drop the run from all three registries.

Without this, a finished run pinned its orchestrator, its whole `CycleRecord` history, its
stream and every subscriber queue for the lifetime of the process. Retention is counted
rather than timed on purpose: a time-to-live would need a clock sweep to reclaim a quiet
server's runs, whereas a count gives a hard upper bound on memory that does not depend on
how long the process has been up. The trade-off is that an old run holds its slot until 50
newer runs finish, so a long pause between runs keeps the previous one readable. The
`_RUN_TASKS` entry is also dropped in the same done callback, so a finished task does not
stay alive until exit.

The deque is deliberately unbounded rather than a `deque(maxlen=...)`, because a `maxlen`
deque would silently drop ids from the left without evicting them from the registries —
leaking exactly what this mechanism exists to fix. `_prune_finished_runs` bounds its length
instead.

### Replay: a 256-event buffer per run

`_RunStream.history` is a bounded ring of the last `_HISTORY_SIZE = 256` events. One cycle
emits roughly `4 + 2 * len(tasks)` events, so 256 frames is a dozen-odd cycles of replay —
far more than a socket attaching mid-run needs — while keeping a retained run's buffer to a
few hundred KB.

This buffer is load-bearing, not a convenience. `_start_run` schedules the run with
`asyncio.create_task` and only *then* does the `/ws` handler reach `subscribe()`. Meanwhile
`Orchestrator.run()` emits `cycle_start` **before its first `await`**, so the task is already
running by the time the handler subscribes. `publish()` can only fan out to subscribers that
exist at the instant it runs, so without a buffer every real client would lose the opening
frames of every run — deterministically, not intermittently.

`subscribe()` pre-loads its queue from `history` and only then registers the queue as a live
subscriber. Both steps are synchronous and neither awaits, so on a single event loop they
cannot interleave with a `publish`: no event is lost between the replay and the subscription,
and none is delivered twice either, because events published after that point reach the live
queue only. If the run already finished, the replay is followed immediately by the `None`
sentinel, so an attaching client reads the whole history and then closes cleanly.

Together these give two behaviours: a client attaching to an already-finished run can still
read what it missed, and a client attaching early still sees a lossless stream.

## Lint and type policy

All three tool configurations live in `pyproject.toml`, and they are deliberately explicit
rather than relying on defaults.

### ruff

```toml
select = ["E", "F", "W", "I", "UP", "B", "SIM", "C4", "RET", "ARG"]
```

The motivation for pinning an explicit `select` is stated in the config's own comment:
without one, plain `ruff check .` only ran the default `E4,E7,E9,F`, so different people
passed different ad-hoc `--select` flags and reported wildly different counts. Two
reviewers could disagree 15 versus 34 errors about the same commit, and neither number meant
anything. One committed list makes `ruff check .` reproducible for everyone.

The `S` (bandit), `BLE`, `TRY` and `N` families are intentionally **not** selected. This
codebase uses broad `except Exception` and `try/except/pass` as deliberate policy: a broken
event observer must never take the development loop down, and a failed test run must be
reported rather than silently skipped. Enforcing those families produces noise on exactly
the constructs that are intentional, so it would train people to ignore the linter.

Also configured:

- `[tool.ruff.lint.isort]` with `known-first-party = ["furrow"]`, so the package's own
  modules sort as first-party imports.
- `[tool.ruff.lint.per-file-ignores]` ignoring `E501` for `furrow/web/server.py` only. That
  module embeds the entire single-page console — HTML, CSS and JavaScript — in one raw
  string literal, so its long lines are markup rather than Python; reflowing them would only
  change what the browser renders. Every other rule still applies to the file.
- `extend-exclude = ["build"]`, because that directory holds stale copies of the package
  produced by `setup.py build` / `pip install .` which drift from the real sources.

### mypy

```toml
python_version = "3.10"
warn_unused_ignores = true
warn_redundant_casts = true
no_implicit_optional = true
check_untyped_defs = true
ignore_missing_imports = true
exclude = ["^tests/"]
```

`ignore_missing_imports = true` because the third-party SDKs this project calls — `anthropic`,
`openai`, `tenacity` — ship no type information. Their attribute access is left unchecked
rather than papered over with hand-written stubs, which would drift from the real SDKs and
would have to be maintained on every dependency bump.

`exclude = ["^tests/"]` keeps type checking scoped to the `furrow` package.

## Lazy package re-exports

All four of `furrow/__init__.py`, `furrow/core/__init__.py`, `furrow/agents/__init__.py` and
`furrow/web/__init__.py` expose their public names through a PEP 562 module-level
`__getattr__` instead of eager `from ... import ...` statements, with a matching `__dir__`.

This is load-bearing, not stylistic. `furrow/__init__.py` documents the exact cycle:

```text
furrow.llm -> furrow.core.json_utils -> furrow.core -> furrow.core.orchestrator
          -> furrow.agents.planner -> furrow.agents -> back to furrow.llm
```

with an eager import failing as "cannot import name 'LLMClient' from partially initialized
module". Each `__init__.py` repeats the same warning about its own slice of the cycle.

The public surface is unchanged — `furrow.LLMClient`, `from furrow import LLMClient`,
`from furrow.core import Orchestrator`, `furrow.web.app` all work — but the submodule import
is deferred to first attribute access, which breaks the cycle. Do not "simplify" these back
into eager imports.

One packaging detail in the same family: **`furrow/cli/__init__.py` must not be deleted.**
`[tool.setuptools.packages.find]` uses `include = ["furrow*"]`, and `find_packages` only
discovers directories that contain an `__init__.py`. The file is currently present, so
`find_packages(where=".", include=["furrow*"])` resolves to `furrow`, `furrow.cli`,
`furrow.agents`, `furrow.web` and `furrow.core`. Remove that file and `furrow.cli` is
dropped from the built distribution, which breaks the `furrow = "furrow.cli.main:main"`
console script in any non-editable install.

## Design notes and known limitations

Each item below is confirmed by reading the current code.

- **Workers cannot change code.** `WorkerAgent.run` is a single `LLMClient.complete` call
  that returns the model's text. There is no tool loop, no read/edit dispatch and no call to
  `LLMClient.write_file`. A plan is executed and reported, but the workspace is not modified
  by the Python package. This is the largest gap between it and the `.kilo/` workers.
- **`completed` is not "tests passed".** The completion predicate inspects task statuses
  only, and a task is `completed` if the worker call did not raise. A run can finish with
  `run_complete` after a failing test run. See the halt conditions above.
- **No persistence.** All state lives in memory: `Orchestrator.history`, and the
  `_RUNS` / `_RUN_TASKS` / `_STREAMS` / `_FINISHED_RUNS` registries in
  `furrow/web/server.py`. Restarting the process loses every run, and there is no resume.
  `.kilo/FURROW.md` lists a persistent state file under "Future Work".
- **Plans are non-deterministic.** The planner is a single LLM call, so the task split
  varies between runs for the same goal. Nothing seeds or constrains it.
- **Dependencies are not scheduled.** `TaskModel.dependencies` is parsed and carried but
  never used to order execution; workers all run as soon as a slot frees.
- **`workspace` is test-only.** It is read in exactly one place,
  `TesterAgent._run_tests`, as the working directory for the test command. It is not a
  sandbox for workers, which do not touch the filesystem at all today.
- **Settings are still process-global at the CLI.** `furrow/config.py` instantiates
  `settings = Settings()` at import. Neither entry point mutates it any more — `start`
  passes a `model_copy` down to the `Orchestrator` — but every run in a process starts from
  that one object, so there is no per-run isolation of any other setting yet.
