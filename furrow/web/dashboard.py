"""Self-contained HTML dashboard for the Furrow web UI.

The page is assembled from small string constants (stylesheet, static
skeleton, and two halves of the client script) so that no single
constant is unwieldy. ``build_dashboard`` composes them and injects
the WebSocket token query string.

Security: every piece of untrusted text (user goals, model-generated
task descriptions, worker results, test failures) is rendered through
``textContent`` / ``createElement`` -- the client script never touches
``innerHTML``.
"""

from __future__ import annotations

import json
from urllib.parse import urlencode

_CSS_LAYOUT: str = """
:root {
  color-scheme: dark;
  --bg: #0e1014;
  --panel: #161a21;
  --panel-2: #1d222c;
  --border: #2b3240;
  --text: #e7eaf0;
  --dim: #8d96a7;
  --accent: #5b9dff;
  --green: #43d17c;
  --red: #ff6b66;
  --yellow: #f2b93c;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 15px;
  line-height: 1.55;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 0.7rem 1.25rem;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}
.topbar h1 { margin: 0; font-size: 1.15rem; font-weight: 650; letter-spacing: 0.02em; }
.conn {
  font-size: 0.78rem;
  padding: 0.18rem 0.65rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--dim);
  text-transform: lowercase;
  white-space: nowrap;
}
.conn-open { color: var(--green); border-color: var(--green); }
.conn-connecting, .conn-reconnecting { color: var(--yellow); border-color: var(--yellow); }
.conn-error, .conn-closed { color: var(--red); border-color: var(--red); }
main { max-width: 920px; margin: 0 auto; padding: 1.1rem 1.25rem 3rem; }
#start-form { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
#goal {
  flex: 1 1 300px;
  padding: 0.5rem 0.7rem;
  font-size: 0.95rem;
  color: var(--text);
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 6px;
}
#goal:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
button {
  padding: 0.5rem 1.1rem;
  font-size: 0.95rem;
  color: var(--text);
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
}
button:disabled { opacity: 0.4; cursor: not-allowed; }
#start-btn { background: var(--accent); border-color: var(--accent); color: #0b1020; font-weight: 600; }
#stop-btn { color: var(--red); border-color: var(--red); }
.notice { margin: 0.45rem 0 0; min-height: 1.3em; font-size: 0.85rem; color: var(--dim); }
.events {
  margin-top: 1rem;
  padding: 0.7rem 0.8rem;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  max-height: 66vh;
  overflow-y: auto;
}
"""

_CSS_COMPONENTS: str = """
.banner { margin: 0.35rem 0; padding: 0.5rem 0.75rem; border-radius: 6px; font-weight: 600; }
.banner-run { background: rgba(91, 157, 255, 0.14); border-left: 3px solid var(--accent); }
.banner-done { background: rgba(67, 209, 124, 0.12); border-left: 3px solid var(--green); }
.banner-fail { background: rgba(255, 107, 102, 0.14); border-left: 3px solid var(--red); }
.cycle-head { display: flex; gap: 0.6rem; align-items: baseline; margin: 0.95rem 0 0.25rem; }
.cycle-title { font-weight: 700; }
.cycle-state { font-size: 0.82rem; }
.cycle-state.pass { color: var(--green); }
.cycle-state.fail { color: var(--red); }
.task {
  margin: 0.3rem 0;
  padding: 0.35rem 0.6rem;
  border-radius: 0 6px 6px 0;
  border-left: 3px solid var(--border);
  background: rgba(255, 255, 255, 0.025);
}
.task-started { border-left-color: var(--accent); }
.task-completed { border-left-color: var(--green); }
.task-failed { border-left-color: var(--red); }
.task-skipped { border-left-color: var(--yellow); }
.task-head { display: flex; gap: 0.55rem; align-items: baseline; }
.task-badge {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--dim);
}
.task-id { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-weight: 600; }
.task-time { margin-left: auto; font-size: 0.75rem; color: var(--dim); }
.task-desc { margin-top: 0.1rem; }
.task-detail {
  margin-top: 0.15rem;
  font-size: 0.85rem;
  color: var(--dim);
  white-space: pre-wrap;
  word-break: break-word;
}
.task-failed .task-detail { color: var(--red); }
.task-skipped .task-detail { color: var(--yellow); }
.tests {
  margin: 0.35rem 0;
  padding: 0.45rem 0.7rem;
  border: 1px solid var(--border);
  border-left: 3px solid var(--border);
  border-radius: 6px;
}
.tests-pass { border-left-color: var(--green); }
.tests-fail { border-left-color: var(--red); }
.tests-summary { color: var(--dim); }
.failure-list {
  margin: 0.35rem 0 0;
  padding-left: 1.2rem;
  color: var(--red);
  font-size: 0.85rem;
  white-space: pre-wrap;
  word-break: break-word;
}
.plan {
  margin: 0.3rem 0;
  padding: 0.4rem 0.6rem;
  border-left: 3px solid var(--accent);
  border-radius: 0 6px 6px 0;
  background: rgba(91, 157, 255, 0.07);
}
.plan-rationale {
  margin-top: 0.2rem;
  font-size: 0.85rem;
  color: var(--dim);
  white-space: pre-wrap;
  word-break: break-word;
}
.log {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.8rem;
  padding: 0.08rem 0.35rem;
  white-space: pre-wrap;
  word-break: break-word;
}
.log-info { color: var(--dim); }
.log-warn { color: var(--yellow); }
.log-error { color: var(--red); }
.log-time { opacity: 0.65; }
"""

_HEAD: str = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Furrow</title>
<style>
"""

_HEAD_TAIL: str = """</style>
</head>
<body>
<header class="topbar">
  <h1>Furrow</h1>
  <div id="conn" class="conn conn-closed" role="status">closed</div>
</header>
<main>
  <form id="start-form">
    <input id="goal" type="text" autocomplete="off" required
           placeholder="Describe what Furrow should accomplish"
           aria-label="Goal">
    <button id="start-btn" type="submit">Start</button>
    <button id="stop-btn" type="button" disabled>Stop</button>
  </form>
  <p id="notice" class="notice" aria-live="polite"></p>
  <section id="events" class="events" aria-live="polite"></section>
</main>
<script>
"""

_JS_STATE: str = """
"use strict";
(function () {
  // Injected by build_dashboard(): "" or "?token=..." (a JSON string
  // literal, so the value is safely escaped).
  const WS_QUERY = __WS_QUERY__;
  const MAX_RECONNECTS = 6;
  const BASE_DELAY_MS = 500;
  const MAX_DELAY_MS = 30000;
  const MAX_ENTRIES = 400;

  const form = document.getElementById("start-form");
  const goalInput = document.getElementById("goal");
  const startBtn = document.getElementById("start-btn");
  const stopBtn = document.getElementById("stop-btn");
  const notice = document.getElementById("notice");
  const conn = document.getElementById("conn");
  const eventsEl = document.getElementById("events");

  let ws = null;
  let generation = 0;
  let reconnectAttempts = 0;
  let intentionalClose = false;
  let running = false;
  let stopRequested = false;
  let sawTerminal = false;
  let pendingGoal = null;

  function setStatus(text, mode) {
    conn.textContent = text;
    conn.className = "conn conn-" + mode;
  }

  function setNotice(text) {
    notice.textContent = text;
  }

  function resetControls() {
    running = false;
    stopRequested = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    goalInput.disabled = false;
  }

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return proto + "//" + location.host + "/ws" + WS_QUERY;
  }

  function sendStart(goal) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ goal: goal }));
      return true;
    }
    return false;
  }
"""

_JS_NET: str = """
  function connect(goal) {
    const gen = ++generation;
    pendingGoal = goal;
    intentionalClose = false;
    setStatus("connecting", "connecting");
    ws = new WebSocket(wsUrl());

    ws.onopen = function () {
      if (gen !== generation) {
        return;
      }
      reconnectAttempts = 0;
      setStatus("open", "open");
      setNotice("");
      if (pendingGoal !== null) {
        const goalText = pendingGoal;
        pendingGoal = null;
        running = true;
        stopRequested = false;
        sawTerminal = false;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        goalInput.disabled = true;
        sendStart(goalText);
      }
    };

    ws.onerror = function () {
      if (gen !== generation) {
        return;
      }
      setStatus("error", "error");
      setNotice("WebSocket error -- see the Furrow server log");
    };

    ws.onclose = function () {
      if (gen !== generation) {
        return;
      }
      if (intentionalClose) {
        setStatus("closed", "closed");
        return;
      }
      if (sawTerminal || reconnectAttempts >= MAX_RECONNECTS) {
        resetControls();
        setStatus("closed", "closed");
        if (!sawTerminal) {
          setNotice(
            "connection lost -- gave up after " + MAX_RECONNECTS + " attempts"
          );
        }
        return;
      }
      const delay = Math.min(
        BASE_DELAY_MS * Math.pow(2, reconnectAttempts),
        MAX_DELAY_MS
      );
      reconnectAttempts += 1;
      setStatus(
        "reconnecting (attempt " + reconnectAttempts +
          " of " + MAX_RECONNECTS + " in " + delay + "ms)",
        "reconnecting"
      );
      setTimeout(function () {
        if (gen === generation) {
          connect(pendingGoal);
        }
      }, delay);
    };

    ws.onmessage = function (evt) {
      if (gen !== generation) {
        return;
      }
      let frame;
      try {
        frame = JSON.parse(evt.data);
      } catch (err) {
        return;
      }
      handleFrame(frame);
    };
  }

  function requestStart() {
    const goal = goalInput.value.trim();
    if (!goal || running) {
      return;
    }
    startBtn.disabled = true;
    intentionalClose = true;
    if (ws) {
      ws.close();
      ws = null;
    }
    connect(goal);
  }

  function requestStop() {
    if (stopRequested || !running) {
      return;
    }
    stopRequested = true;
    stopBtn.disabled = true;
    setNotice("stop requested -- waiting for the run to wind down");
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "stop" }));
    }
  }

  form.addEventListener("submit", function (evt) {
    evt.preventDefault();
    requestStart();
  });
  stopBtn.addEventListener("click", requestStop);
  window.addEventListener("beforeunload", function () {
    intentionalClose = true;
    if (ws) {
      ws.close();
    }
  });

  window.handleFrame = function (frame) {
    if (!frame || typeof frame.type !== "string") {
      return;
    }
    if (frame.type === "run.completed" || frame.type === "run.failed") {
      sawTerminal = true;
      resetControls();
    }
    window.renderEvent(frame);
    while (eventsEl.childElementCount > MAX_ENTRIES) {
      eventsEl.removeChild(eventsEl.firstElementChild);
    }
    eventsEl.scrollTop = eventsEl.scrollHeight;
  };

  connect(null);
})();
"""

_JS_RENDER_A: str = """
(function () {
  "use strict";

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function text(node, content) {
    node.appendChild(document.createTextNode(String(content)));
    return node;
  }

  function stamp(frame) {
    if (typeof frame.timestamp !== "number") {
      return "";
    }
    return new Date(frame.timestamp * 1000).toLocaleTimeString();
  }

  function cycleOf(frame) {
    return typeof frame.cycle === "number" ? frame.cycle : 0;
  }

  function dataOf(frame) {
    return frame.data && typeof frame.data === "object" ? frame.data : {};
  }

  function renderRunStarted(frame, data) {
    const banner = el("div", "banner banner-run");
    text(banner, "Run started");
    if (data.goal) {
      text(banner, " -- goal: ");
      banner.appendChild(el("span", "goal-text", data.goal));
    }
    if (typeof data.max_cycles === "number" && data.max_cycles > 0) {
      text(banner, " (max " + data.max_cycles + " cycles)");
    }
    return banner;
  }

  function renderRunCompleted(frame, data) {
    const banner = el("div", "banner banner-done");
    text(banner, "Run completed");
    const cycles = typeof data.cycles === "number" ? data.cycles : 0;
    text(banner, " after " + cycles + " cycle(s)");
    if (data.reason) {
      text(banner, ": " + data.reason);
    }
    return banner;
  }

  function renderRunFailed(frame, data) {
    const banner = el("div", "banner banner-fail");
    text(banner, "Run failed");
    if (data.error) {
      text(banner, " -- " + data.error);
    }
    return banner;
  }

  function renderCycleStarted(frame, data) {
    const head = el("div", "cycle-head");
    head.appendChild(el("span", "cycle-title", "Cycle " + cycleOf(frame)));
    if (typeof data.cycle === "number") {
      head.appendChild(el("span", "cycle-state", "started"));
    }
    return head;
  }

  function renderCycleCompleted(frame, data) {
    const head = el("div", "cycle-head");
    head.appendChild(
      el("span", "cycle-title", "Cycle " + cycleOf(frame) + " finished")
    );
    const passed = !!data.passed;
    head.appendChild(
      el("span", passed ? "cycle-state pass" : "cycle-state fail",
         passed ? "passed" : "failed")
    );
    return head;
  }

  function renderPlanCreated(frame, data) {
    const plan = data.plan && typeof data.plan === "object" ? data.plan : {};
    const tasks = Array.isArray(plan.tasks) ? plan.tasks : [];
    const block = el("div", "plan");
    text(block, "Plan created: " + tasks.length + " task(s)");
    if (plan.rationale) {
      block.appendChild(el("div", "plan-rationale", plan.rationale));
    }
    return block;
  }
"""

_JS_RENDER_B: str = """
  function renderTask(frame, data, kind, label, detail) {
    const row = el("div", "task task-" + kind);
    const head = el("div", "task-head");
    head.appendChild(el("span", "task-badge", label));
    head.appendChild(el("span", "task-id", data.task_id || "?"));
    const time = stamp(frame);
    if (time) {
      head.appendChild(el("span", "task-time", time));
    }
    row.appendChild(head);
    if (data.description) {
      row.appendChild(el("div", "task-desc", data.description));
    }
    if (detail) {
      row.appendChild(el("div", "task-detail", detail));
    }
    if (kind === "started" && Array.isArray(data.files) && data.files.length) {
      row.appendChild(
        el("div", "task-detail", "files: " + data.files.join(", "))
      );
    }
    return row;
  }

  function renderTestsCompleted(frame, data) {
    const passed = !!data.passed;
    const block = el("div", passed ? "tests tests-pass" : "tests tests-fail");
    block.appendChild(
      el("strong", null, passed ? "Tests passed" : "Tests failed")
    );
    if (data.summary) {
      text(block, " -- " + data.summary);
    }
    if (!passed && Array.isArray(data.failures) && data.failures.length) {
      const list = el("ul", "failure-list");
      for (const failure of data.failures) {
        list.appendChild(el("li", null, failure));
      }
      block.appendChild(list);
    }
    return block;
  }

  function renderLog(frame, data) {
    const level = String(data.level || "info").toLowerCase();
    const cls = level === "error" ? "log-error"
      : level === "warning" ? "log-warn" : "log-info";
    const line = el("div", "log " + cls);
    const time = stamp(frame);
    if (time) {
      line.appendChild(el("span", "log-time", time + " "));
    }
    line.appendChild(el("span", "log-msg", data.message));
    return line;
  }

  window.renderEvent = function (frame) {
    const data = dataOf(frame);
    let node;
    switch (frame.type) {
      case "run.started":
        node = renderRunStarted(frame, data);
        break;
      case "run.completed":
        node = renderRunCompleted(frame, data);
        break;
      case "run.failed":
        node = renderRunFailed(frame, data);
        break;
      case "cycle.started":
        node = renderCycleStarted(frame, data);
        break;
      case "cycle.completed":
        node = renderCycleCompleted(frame, data);
        break;
      case "plan.created":
        node = renderPlanCreated(frame, data);
        break;
      case "task.started":
        node = renderTask(frame, data, "started", "task", "");
        break;
      case "task.completed":
        node = renderTask(
          frame, data, "completed", "done",
          "result: " + (data.result || "") +
            (typeof data.duration === "number"
              ? " (" + data.duration.toFixed(2) + "s)" : "")
        );
        break;
      case "task.failed":
        node = renderTask(
          frame, data, "failed", "failed",
          "error: " + (data.error || "unknown error") +
            (typeof data.duration === "number"
              ? " (" + data.duration.toFixed(2) + "s)" : "")
        );
        break;
      case "task.skipped":
        node = renderTask(
          frame, data, "skipped", "skipped",
          "reason: " + (data.reason || "")
        );
        break;
      case "tests.completed":
        node = renderTestsCompleted(frame, data);
        break;
      case "log":
        node = renderLog(frame, data);
        break;
      default:
        node = el("div", "log log-info", "[" + frame.type + "]");
        break;
    }
    document.getElementById("events").appendChild(node);
  };
})();
"""

_TAIL: str = """</script>
</body>
</html>
"""


def build_dashboard(token: str | None = None) -> str:
    """Compose the dashboard page, embedding the WebSocket token.

    ``token`` is the ``FURROW_WEB_TOKEN`` value the server gates the
    ``/ws`` endpoint with; when set, it is appended to the WebSocket
    URL as the ``token`` query parameter so the client can
    authenticate. ``json.dumps`` produces a safe JS string literal,
    and ``urlencode`` percent-escapes the value.
    """
    query = "?" + urlencode({"token": token}) if token else ""
    head_js = _JS_STATE.replace("__WS_QUERY__", json.dumps(query))
    return (
        _HEAD
        + _CSS_LAYOUT
        + _CSS_COMPONENTS
        + _HEAD_TAIL
        + head_js
        + _JS_NET
        + _JS_RENDER_A
        + _JS_RENDER_B
        + _TAIL
    )
