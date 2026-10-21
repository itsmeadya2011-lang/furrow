# Furrow Improvement Opportunities

Based on the codebase and the explicit Future Work in `.kilo/FURROW.md`, here are prioritized concrete recommendations.

## 1) Cost accounting (currency) — High impact, Low-Medium effort
**Aligns with:** "Cost accounting: currency cost estimation is still missing" in Future Work. Also README notes only token counts tracked.

**Gaps found:**
- `Usage` only tracks `input_tokens`, `output_tokens`, `calls`. No pricing model or cost calculation.
- `LLMClient.total_usage` accumulates token counts but has no concept of cost; different models/providers have different pricing.
- State/result serialization includes usage but not cost. No cost surfaced in web UI except usage counts.
- `Settings` has no pricing configuration.

**Recommendations (concrete):**
- Extend `Usage` with `cost_usd: float = 0.0` and a `cost()` method; track cost per call/response via pricing tables. Add `Usage.add_cost()` or accumulate in `LLMClient` using `Settings`-based pricing.
- Add pricing config (e.g., `FURROW_PRICING_*` or a model→(input_per_1k, output_per_1k)) supporting per-model override; allow custom pricing. Include provider defaults for common models.
- Compute cost in `LLMClient._complete_*` / `_chat_*` using response usage + model, add to `total_usage.cost_usd`. Expose cost in `RunResult`, `HistoryEntry`, `RunState`, and print in CLI/web.
- Show estimated cost in CLI summary and web UI header (alongside tokens/calls). Add `cost` to `/api/status`.

**Expected impact:** High — essential for real deployments (users need spend visibility).  
**Effort:** Low-Medium (schema additions + pricing tables + plumbing).

## 2) Richer web UI with per-agent panels — High impact, Medium effort
**Aligns with:** "Richer web UI with per-agent panels" in Future Work.

**Gaps found:**
- Web UI currently shows tasks list with status/attempts/files/result/error (generic). No explicit "planner/worker/tester" agent identity, per-agent logs, timing, or usage split by agent.
- Events include agent activity (`TASK_STARTED/FINISHED`, `TEST_STARTED/FINISHED`, `PLAN_CREATED`) but UI doesn't group by agent or show agent panels.
- No real-time streaming of per-agent thoughts/tool calls in a structured panel view; just a log stream.
- No visibility into which worker is doing what in parallel waves.

**Recommendations (concrete):**
- Track agent identity in events (e.g., add `agent` field to `Event` or include in `data` like `{"agent":"worker","task":"1"}`). Extend `TASK_STARTED`/worker events to include agent name.
- Add per-agent state to API (`/api/agents`) or include in status: planner/worker instances, current task, iterations, tool calls, files changed, elapsed time.
- Redesign UI with panels per agent type/instance (Planner panel, Worker #1/#2..., Tester). Show live status, last action, usage/cost per agent, and expandable details.
- Add filters, wave visualization, and better task cards with dependencies/files. Consider collapsing logs into agent-specific streams.

**Expected impact:** High — much better observability for debugging long-running runs.  
**Effort:** Medium (event schema expansion + backend aggregation + frontend rewrite).

## 3) Richer `furrow` package plugin system — High impact, Medium-High effort
**Aligns with:** "A richer `furrow` package plugin system" in Future Work.

**Gaps found:**
- No plugin/extension hooks visible in codebase (no `plugins/`, no entry points). Components are hardwired: `PlannerAgent`, `WorkerAgent`, `TesterAgent`, `Orchestrator`, `EventBus`, `LLMClient` instantiated directly.
- Tools for worker are fixed in `prompts.WORKER_TOOLS` and handled in `WorkerAgent._execute_tool`. No way to register custom tools.
- Test detection in `TesterAgent` is hardcoded per project type. No extensibility.
- EventBus exists but no plugin lifecycle/hooks (pre/post plan/execute/test, custom validators).

**Recommendations (concrete):**
- Define plugin API (e.g., `FurrowPlugin` base class with hooks: `on_cycle_start`, `on_plan_created`, `on_task_execute`, `on_test_finished`, `register_tools()`, `detect_tests()`). Support discovery via entry points (`furrow.plugins`) or local `.furrow/plugins/`.
- Make components pluggable: allow custom planner/worker/tester classes via config/settings; add tool registry so plugins can add tools to worker with sandboxing/permissions.
- Add plugin manager to orchestrator; emit plugin events; provide examples and docs.
- Ensure plugin failures are isolated (don't crash run).

**Expected impact:** High — enables extensibility (custom tools, CI integrations, domain-specific agents).  
**Effort:** Medium-High (API design + lifecycle + loading + security/sandboxing).

## 4) Multi-provider model routing — High impact, Medium effort
**Aligns with:** "Multi-provider model routing" in Future Work.

**Gaps found:**
- Settings have single provider/model plus separate `planner_model`, `worker_model`, `tester_model` but all use the same `provider`. No per-role/provider routing or fallback chains.
- `LLMClient` uses single `self.settings.provider` for all calls. No load balancing, cost/latency based routing, or fallback on errors.
- No way to route different tasks/agents to different providers/models based on rules (e.g., cheap model for simple tasks, capable for complex).
- Retry logic exists per call but no cross-provider fallback.

**Recommendations (concrete):**
- Extend config: allow per-role provider+model+API key (e.g., `FURROW_PLANNER_PROVIDER`, fallbacks), and routing rules (by task complexity/files/length). Support model lists with priority/fallback.
- Refactor `LLMClient` to a router that selects provider/client based on role/model/routing policy, with health tracking and automatic fallback on retryable failures.
- Add routing strategies: round-robin, cheapest-first (using pricing), latency-based. Respect API key availability.
- Track per-provider usage/cost separately.

**Expected impact:** High — resilience, cost optimization, access to best models per task.  
**Effort:** Medium (config + router + fallback logic + tests).

## 5) Worker/tools and robustness improvements — High impact, Low-Medium effort
**Gaps found (from code inspection):**
- **Tool set limited:** Worker only has `read_file`, `write_file`, `list_files`, `run_command`. No `search`/`grep`, `edit` (patch), `glob`, `delete`, `move/copy`, or git tools. This limits efficiency vs. what's possible in general coding agents.
- **Path safety:** `resolve_in_workspace` is good (prevents escapes). But `list_files` uses `rglob("*")` and caps at 500 entries (good). `read_file` truncates at 20KB (good). No size limits on write_file content.
- **Command execution:** Denylist exists but is very small (e.g., `rm -rf /`, `mkfs`, etc.). No allowlist, no cwd restrictions beyond workspace, no environment sanitization. Subprocess uses `shlex.split` (good) but doesn't prevent shell metacharacters if parsed oddly; also doesn't enforce max runtime env limits.
- **Error handling:** Worker catches exceptions broadly in `_run()` returning `WorkerResult` with error (good). Orchestrator catches task exceptions and retries (good). But some errors may not be surfaced clearly; tool errors returned as strings to LLM (good pattern). LLM retry logic exists.
- **No tool result size limits:** `run_command` output truncated to 20KB (good). `read_file` truncated to 20KB (good). Tool results go back to LLM - need to ensure all tools respect limits.
- **Iteration budget:** Correctly fails (not succeeds) on exhaustion (tested). Good.
- **Process group kill:** Implemented for POSIX (good, tested). Fallback exists.
- **No streaming/tool observability:** Hard to debug long tool calls.
- **File operations:** No atomic writes in worker's write_file path? Worker uses `client.write_file` which does `mkdir(parents=True, exist_ok=True)` and writes directly (overwrites). State uses atomic writes; worker should consider atomic writes for critical files.
- **Testing gaps:** Some areas could use more coverage (e.g., plugin system none, cost accounting none, multi-provider routing none, web UI edge cases, concurrent task failure propagation edge cases).

**Concrete recommendations:**
- **Add useful tools:** `grep_search`/`glob` (using ripgrep if available, else Python), `apply_patch`/`edit_file` (safer than rewriting whole files), `git` tools (status/diff/commit) - but need to be careful about permissions. Add to `WORKER_TOOLS` and implement in `WorkerAgent`.
- **Strengthen command safety:** Expand denylist, add allowlist mode, set env (PATH minimal), run with `ulimit` if possible, add `FURROW_ALLOW_COMMANDS`/blocklists. Consider `sandbox` libraries or at least better validation.
- **Improve limits:** Add `max_write_size`, `max_command_output` (configurable), enforce in tools. Truncate more aggressively with clear messages.
- **Better error reporting:** Include tool name/args in errors (without leaking secrets), structured errors. Add error classification.
- **Atomic writes in worker:** Use temp file + rename in worker's file operations to avoid partial writes on crash.
- **Tool timeouts per tool:** Different timeouts for read/write vs commands.
- **Add tests:** More tests for edge cases (concurrent failures, retry exhaustion, state corruption recovery, web API race conditions, process cleanup on various signals).

**Expected impact:** High — better capability, safety, reliability.  
**Effort:** Low-Medium (incremental additions).

## Summary (Prioritized)

| Priority | Improvement | Impact | Effort | Notes |
|----------|-------------|--------|--------|-------|
| 1 | Cost accounting (currency) | High | Low-Med | Explicitly called out as missing; needed for production use. Easy to bolt onto existing Usage. |
| 2 | Richer web UI with per-agent panels | High | Med | Improves UX/debugging significantly; aligns with stated direction. |
| 3 | Multi-provider model routing | High | Med | Resilience + cost control; natural extension of existing multi-model support. |
| 4 | Richer plugin system | High | Med-High | Long-term extensibility; bigger architectural change. |
| 5 | Worker/tools + robustness | High | Low-Med | Immediate capability/safety gains; can be done incrementally. |

**Top recommendation:** Start with **cost accounting (currency)** - it's explicitly listed as missing, has clear extension points in `Usage`/`LLMClient`/state/CLI/web, and provides immediate user value with relatively contained changes.
