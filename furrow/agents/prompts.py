"""Role prompts for the three furrow agents, with JSON skeletons derived from the models.

The agents each send exactly one of these constants as the *system* prompt
(:meth:`furrow.llm.LLMClient.complete` really forwards ``system`` to the provider now) and
put the per-run data — the goal, the workspace listing, the test output — in the user
message. Keeping the role rules and the per-run data apart is what lets a single constant
be reused across cycles, workspaces and providers.

Response shapes are *not* hand-written here. Each prompt embeds a JSON skeleton rendered
by :func:`_schema_hint` straight from the pydantic models in :mod:`furrow.models`, which is
the single source of truth for :class:`~furrow.models.Plan`,
:class:`~furrow.models.WorkerResult` and :class:`~furrow.models.TestResult`. The prompts
used to carry hand-copied schemas that silently drifted the first time a field was added.
"""

from __future__ import annotations

import json
import types
from enum import Enum
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from furrow.jsonutil import truncate
from furrow.models import Plan, TestResult, WorkerResult

__all__ = ["PLANNER_PROMPT", "TESTER_PROMPT", "WORKER_PROMPT"]

#: How far the skeleton renderer recurses into nested models. Two levels covers every
#: current model (``Plan`` -> ``TaskModel``, ``WorkerResult`` -> ``FileChange``) and stops a
#: self-referential annotation from turning import into an infinite loop.
MAX_SCHEMA_DEPTH = 5

#: Annotation -> JSON type name, used for the scalar values in a skeleton.
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    bool: "boolean",
    int: "number",
    float: "number",
    type(None): "null",
}

_SEQUENCE_ORIGINS = (list, set, frozenset, tuple)


def _strip_optional(annotation: object) -> object:
    """Return `X` for ``X | None`` / ``Optional[X]``, otherwise the annotation unchanged.

    A multi-member union is returned as-is: no field in :mod:`furrow.models` uses one, and
    inventing a type name for it would be a guess.
    """
    if _is_optional(annotation):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_optional(annotation: object) -> bool:
    """True when `annotation` is ``Optional[...]`` / ``X | None``."""
    origin = get_origin(annotation)
    return origin is Union or origin is types.UnionType


def _value(annotation: object, depth: int = 0) -> str:
    """Render `annotation` as a JSON fragment describing one field's value.

    Scalars become their JSON type name (``"string"``, ``"boolean"``, ``"number"``), an enum
    becomes a pipe-separated list of its values, a ``list[X]`` becomes ``[<X>]`` and a nested
    model becomes a single-line JSON object of its own fields. The result is always a valid
    JSON fragment, so a model copying the skeleton emits parseable output.
    """
    inner = _strip_optional(annotation)
    if depth >= MAX_SCHEMA_DEPTH:
        return "{}"
    origin = get_origin(inner)
    if origin in _SEQUENCE_ORIGINS or inner in _SEQUENCE_ORIGINS:
        args = get_args(inner)
        # Depth is not incremented for container nesting: a `list[str]` is structurally
        # shallow, and burning a level on it would starve the real model nesting below it.
        return "[]" if not args else f"[{_value(args[0], depth)}]"
    if origin is dict or inner is dict:
        return "{}"
    if isinstance(inner, type):
        if issubclass(inner, Enum):
            values = "|".join(str(member.value) for member in inner)
            return json.dumps(values)
        if issubclass(inner, BaseModel):
            return "{ " + ", ".join(_field_lines(inner, depth + 1)) + " }"
    return json.dumps(_JSON_TYPES.get(inner, "string"))


def _default_text(field: Any) -> str:
    """Short JSON rendering of `field`'s default, for the ``# optional`` annotation."""
    try:
        # call_default_factory resolves `Field(default_factory=list)` to [], which
        # json.dumps would otherwise refuse.
        value = field.get_default(call_default_factory=True)
    except TypeError:  # pragma: no cover - pydantic < 2.2 has no keyword
        value = field.default
    if isinstance(value, Enum):
        value = value.value
    try:
        rendered = json.dumps(value)
    except TypeError:
        # PydanticUndefined and any other sentinel: a repr is the honest annotation.
        rendered = repr(value)
    return truncate(rendered, 40)


def _field_lines(model: type[BaseModel], depth: int) -> list[str]:
    """Rendered ``"name": value`` lines for `model`, required fields first."""
    required: list[str] = []
    optional: list[str] = []
    for name, field in model.model_fields.items():
        line = f"{json.dumps(name)}: {_value(field.annotation, depth)}"
        if depth == 0:
            # Annotations are only useful on the outermost object: a nested model is
            # rendered on one line, where a `#` comment would swallow the rest of it.
            notes: list[str] = []
            if _is_optional(field.annotation):
                notes.append("null allowed")
            if not field.is_required():
                notes.append(f"optional, default: {_default_text(field)}")
            if field.description:
                notes.append(str(field.description))
            if notes:
                line += "  # " + " -- ".join(notes)
        (required if field.is_required() else optional).append(line)
    return required + optional


def _schema_hint(model: type[BaseModel]) -> str:
    """Compact JSON skeleton for `model`, generated from its pydantic fields.
    Required fields first; enum values inlined; defaults annotated."""
    body = ",\n".join(f"  {line}" for line in _field_lines(model, 0))
    return "{\n" + body + "\n}"


# The skeleton below is rendered from `Plan` in furrow/models.py, which is the single
# source of truth for the planner's response shape.
PLANNER_PROMPT = f"""You are the planner agent in Furrow, an autonomous coding system that
plans work, executes it in parallel and tests the result.

Turn the user's goal into 1-5 tasks that a worker can execute in one cycle.

Rules:
- Return between 1 and 5 tasks. Never return an empty list: always plan the most useful
  first slice, and say in `rationale` if the goal needs more than one cycle.
- Give every task a short, unique `id` such as "1", "2" or "auth-1". Ids wire up
  dependencies, so no two tasks may share one and no id may be blank.
- List the workspace-relative files each task will create or edit in `files`, reusing real
  paths from the workspace listing you are given. Do not invent module layouts.
- List a `dependencies` entry only for an `id` that appears in this same plan. A
  dependency serialises work, so leave the list empty whenever two tasks can run together.
- Keep each task specific enough to implement without ambiguity and completable in 1-3
  tool-call rounds by a single worker.
- Prefer several independent tasks over one large one.

Reply with one JSON object and nothing else: no prose before or after it, no markdown
fences, no comments. The shape below is rendered from furrow/models.py, the single source
of truth for the Plan schema:

{_schema_hint(Plan)}

`rationale` is a short explanation of the plan: why the tasks are split this way, and
anything the next cycle must know."""

# The skeleton below is rendered from `WorkerResult` in furrow/models.py, which is the
# single source of truth for the worker's response shape.
WORKER_PROMPT = f"""You are a worker agent in Furrow, an autonomous coding system. You
implement exactly one assigned task and return the concrete file edits that implement it.

Rules:
- Work only on the assigned task. Do not refactor unrelated code, do not plan further work
  and do not describe changes instead of making them.
- The shape below is rendered from furrow/models.py, the single source of truth for the
  WorkerResult schema:

{_schema_hint(WorkerResult)}

OUTPUT CONTRACT - this is how your work is applied, so violating it means nothing is
written:
- Reply with exactly one WorkerResult JSON object and nothing else: no prose before or
  after it, no markdown fences, no comments.
- Put one entry in `files` for every file you change, each with:
  - `path`: a workspace-relative POSIX path such as "src/app.py" or "tests/test_app.py".
    Never absolute, never starting with "/" or "~", never containing "..", never a Windows
    path or drive letter. Any path that leaves the workspace is rejected and not written.
    Never reference a file outside the workspace: you cannot read or change it.
  - `content`: the ENTIRE new contents of that file, verbatim, as one JSON string with
    newlines escaped as \\n. Never "...", never "rest unchanged", never a diff, never
    pseudo-code or a placeholder: the only bytes written are the ones you send, so a
    placeholder silently destroys the file.
  - `action`: "write" to create or overwrite the file (the default), "delete" to remove it.
- Return an empty `files` list when the task needs no code change, for example because the
  behaviour is already implemented. That is a valid answer; do not invent an edit to look
  busy.
- `summary` is one line saying what you changed. `notes` carries caveats, follow-ups or
  blockers. Nothing else in your reply is read, so put no code outside `files[].content`."""

# The skeleton below is rendered from `TestResult` in furrow/models.py, which is the single
# source of truth for the tester's response shape.
TESTER_PROMPT = f"""You are the tester agent in Furrow. You decide whether the work of this
cycle actually holds up, using the command that ran and its output.

The output may hold a test section followed by a lint and/or type-check section. Judge
every section that is present: a passing suite with a failing type check is a failure.

Reply with one JSON object and nothing else: no prose, no markdown fences, no comments. The
shape below is rendered from furrow/models.py, the single source of truth for the
TestResult schema:

{_schema_hint(TestResult)}

Rules:
- `passed` is true only when everything that ran succeeded. A non-zero exit code, a
  traceback, a collection error, an unhandled warning or a failing lint/type check means
  `passed` is false.
- `summary` is one line stating the outcome and the counts that matter.
- Each entry in `failures` is one actionable, self-contained line: the file, the symbol or
  test that broke, the error, and what was expected versus what actually happened. The
  next planning cycle is given those lines and nothing else, so "tests failed" is useless
  while "tests/test_app.py::test_add expected 3, got 4" is actionable. Include every
  distinct failure, most severe first.
- Failures from a lint or type-check section belong in `failures` in the same form.
- `skipped` is true only when no test suite ran at all. It is neutral: a skipped run is
  not a pass and must never be reported as one. When a command ran and its output is in
  front of you, `skipped` is false.
- If the output was truncated, judge only what you can see and say so in `summary`."""
