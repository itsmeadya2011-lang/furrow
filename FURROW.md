# Furrow

Furrow is an autonomous coding agent that plans, implements, and tests changes
to a codebase through an iterative loop. It coordinates three LLM-driven
roles — a **planner**, a **worker** (which edits files using real tool calls),
and a **tester** — to move toward a goal while persisting state and optionally
streaming progress via a web UI.

## Architecture

```
CLI / Web
   │
   ▼
Orchestrator ──► Planner ──► Worker ──► Tester
   │                (plan)    (edit)    (verify)
   │
   ▼
EventBus (streams events to the web UI)
StateStore (atomic persistence of run state)
```

The orchestrator runs cycles until the goal is complete, the maximum number
of cycles is reached, or no progress is made.  Each cycle:

1. The planner breaks the goal into 1–5 parallelizable tasks.
2. The worker implements each task using `read_file`, `write_file`,
   `list_files`, and `run_command` (sandboxed to the workspace).
3. The tester detects a test runner (pytest, npm, cargo, go, ...) and runs it;
   the exit status is the ground truth.

## Security model

- The worker runs commands through `/bin/sh -c` in its own process group.
- A hardened command guard (`is_command_denied`) checks every segment of a
  command against a deny-list of patterns, denied program names, force-push
  patterns, command substitutions, and redirections that escape the
  workspace.
- File paths are sandboxed to the workspace via `resolve_in_workspace`.
- The tester routes its detected test command through the same guard.

## Roadmap

### Implemented

- [x] Iterative plan → implement → test loop with persisted state
- [x] Provider-neutral LLM client (Anthropic, OpenAI, Ollama)
- [x] Sandboxed worker tools with a hardened command guard
- [x] Project-aware test detection (pytest, npm/pnpm/yarn, cargo, go)
- [x] Exit-status-is-ground-truth test verdict
- [x] Event bus with bounded queues and a streaming web UI
- [x] Atomic state persistence with resume support
- [x] No-progress guard (stops the loop when a cycle makes no progress)
- [x] Token usage tracking and cost estimation (USD)

### Near term

- [ ] **Currency cost estimation.** Token accounting is delivered; the
      `cost_usd` field is populated from a pricing table, but the table is
      not yet surfaced in the CLI/web summary.  Add a `--cost` flag to the
      CLI and a cost line to the web UI.
- [ ] **Multi-provider model routing.** Let the planner, worker, and tester
      use different providers (e.g. Anthropic for the worker, Ollama for the
      planner).  Currently `Settings` has one `provider` for all roles.
- [ ] **Richer plugin system.** Allow third-party tools and test runners to
      be registered without editing the core.

### Medium term

- [ ] **Parallel task waves.** Tasks within a wave already run concurrently,
      but the orchestrator re-plans after every cycle.  A "commit" mode
      that batches several waves before re-planning would reduce API costs.
- [ ] **Structured task results.** The worker returns a free-text summary.
      A structured result (files changed, tests added, decisions made) would
      let the planner build better context for the next cycle.
- [ ] **Diff review.** A `git diff` view in the web UI so a human can review
      what the worker changed before the tester runs.

### Long term

- [ ] **Self-healing.** When the tester reports a failure, the loop already
      feeds the failure back to the planner.  A dedicated "fixer" role that
      analyses the failing test and proposes a targeted patch would close the
      loop faster than the general planner.
- [ ] **Multi-workspace.** Some goals span multiple repositories.  The
      orchestrator currently assumes a single workspace.
- [ ] **Agent memory.** A vector-indexed memory of past runs so a resumed
      session can recall what it tried before and why it failed.

## Configuration

All settings are prefixed with `FURROW_` and can be supplied via environment
variables or a `.env` file.  See `furrow/config.py` for the full list.

## Development

```bash
pip install -e ".[dev]"
pytest
```