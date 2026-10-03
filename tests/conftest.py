"""Shared fixtures and stubs for the Furrow test suite.

No test may contact an LLM provider. The autouse :func:`no_network_egress`
fixture makes that a hard guarantee rather than a convention: any attempt to
open a socket to anything other than the loopback interface raises
:class:`NetworkAccessBlocked`. Loopback stays usable so a test may still bind a
local port; everything else fails loudly with a message pointing here. Stub the
SDK client objects (``StubLLMClient`` below, or ``LLMClient._anthropic`` /
``._openai`` / ``._ollama``) instead of letting the real ones construct a
request, and the suite passes with no ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``
in the environment and no outbound socket.

Everything that touches the filesystem is pointed at pytest's ``tmp_path``:
:func:`settings_factory` builds a :class:`~furrow.config.Settings` whose
workspace and ``state_file`` live inside the test's own temporary directory and
whose ``.env`` file reading is disabled, so a test can never write into the
repository or inherit an ambient ``FURROW_*`` value by accident.

Fixtures provided
-----------------
``no_network_egress`` (autouse)
    Block non-loopback network access for the duration of every test.
``tmp_workspace``
    ``tmp_path / "workspace"``, created empty, for tests that write files.
``settings_factory``
    ``settings_factory(workspace=None, **overrides) -> Settings`` -- a cheap
    ``Settings`` rooted in ``tmp_path`` with an isolated state file. Any
    ``Settings`` field may be passed as a keyword; ``state_file=None`` falls
    back to ``<workspace>/.furrow/state.json``.
``clean_env``
    Strips every ``FURROW_*`` env var plus the provider keys, so ``Settings()``
    reads its own defaults.
``task_factory`` / ``plan_factory``
    ``make_task`` / ``make_plan`` (see below) for terse model fixtures.
``stub_llm`` / ``stub_llm_factory``
    A ready-made :class:`StubLLMClient`, and a factory for scripted ones.
``scratch_workspace``
    A shared throwaway directory (outside the repo) for stubs built without
    explicit settings.

Module-level helpers (importable as ``from tests.conftest import ...``)
---------------------------------------------------------------------
``StubLLMClient``
    Offline stand-in for :class:`~furrow.llm.LLMClient`. Subclasses it, so it
    can be handed to ``PlannerAgent`` / ``WorkerAgent`` / ``TesterAgent`` /
    ``Orchestrator`` wherever a real client is accepted. Records every call in
    ``.calls`` (``LLMCall(prompt, system, model)``), exposes ``.prompts`` /
    ``.models`` / ``.remaining``, replays a queue of replies (strings, or
    exceptions to raise), then falls back to ``responder(prompt)`` or
    ``default``; ``.queue(...)`` appends more replies mid-test.
``make_task`` / ``make_plan`` / ``plan_json`` / ``verdict_json``
    Concise builders for models and for the JSON strings a model would return
    (``plan_json`` emits a markdown-fenced plan, ``verdict_json`` a bare
    verdict). Keep helper names free of a leading ``test_``: pytest collects any
    module-level callable named ``test_*`` when it is imported into a test
    module.
``offline_settings(**values)``
    ``Settings`` with the ambient ``.env`` lookup disabled.
``stub_request()`` / ``stub_response(status_code)``
    Offline ``httpx`` objects for building SDK error types in retry tests.
``no_retry_sleep()``
    Context manager that zeroes tenacity's backoff sleep for the block, so
    retry tests assert the attempt budget instead of the wall clock.
``fake_api_key()``
    An obviously fake credential for constructing ``Settings`` offline.
"""

from __future__ import annotations

import functools
import ipaddress
import json
import os
import socket
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from httpx import Request as HttpxRequest
from httpx import Response as HttpxResponse

from furrow.config import Plan, Provider, Settings, TaskModel, TaskStatus, TestResult
from furrow.llm import LLMClient

__all__ = [
    "LLMCall",
    "NetworkAccessBlocked",
    "PlanFactory",
    "Responder",
    "SettingsFactory",
    "StubFactory",
    "StubLLMClient",
    "StubResponse",
    "StubResponses",
    "TaskFactory",
    "clean_env",
    "fake_api_key",
    "make_plan",
    "make_task",
    "no_network_egress",
    "offline_settings",
    "plan_factory",
    "plan_json",
    "scratch_workspace",
    "settings_factory",
    "stub_llm",
    "stub_llm_factory",
    "stub_request",
    "stub_response",
    "task_factory",
    "tmp_workspace",
    "verdict_json",
]

# --------------------------------------------------------------------------- #
# Offline guarantee
# --------------------------------------------------------------------------- #


class NetworkAccessBlocked(RuntimeError):
    """Raised when a test tries to open a non-loopback socket."""


_LOOPBACK_NAMES = frozenset(
    {
        "",
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }
)


def _is_local_host(host: object) -> bool:
    """True for hosts that cannot leave this machine.

    Loopback names and addresses pass, and so do ``None`` / the unspecified
    address (the kernel resolves both to this host) so a test may still bind a
    local server the way ``uvicorn.run(app, port=...)`` does.
    """
    if host is None:
        return True
    if not isinstance(host, str):
        return False
    if host in _LOOPBACK_NAMES:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _is_local_address(address: Any) -> bool:
    """True for loopback and unix-domain addresses, which cannot leave the host."""
    if isinstance(address, str):
        # AF_UNIX socket path.
        return True
    if isinstance(address, (tuple, list)) and address:
        return _is_local_host(address[0])
    return False


def _describe(address: Any) -> str:
    if isinstance(address, (tuple, list)) and address:
        return ":".join(str(part) for part in address)
    return str(address)


def _reject(address: Any) -> None:
    if _is_local_address(address):
        return
    raise NetworkAccessBlocked(
        f"blocked outbound network access to {_describe(address)!r}: the Furrow "
        "test suite must stay offline. Stub the LLM client instead "
        "(tests/conftest.py: StubLLMClient, or LLMClient._anthropic / "
        "._openai / ._ollama)."
    )


def _reject_name(name: object) -> None:
    if _is_local_host(name):
        return
    raise NetworkAccessBlocked(
        f"blocked DNS lookup for {name!r}: the Furrow test suite must stay "
        "offline. Use 127.0.0.1, ::1 or 'localhost' if you need a local socket."
    )


@pytest.fixture(autouse=True)
def no_network_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything in a test tries to leave the loopback interface."""
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    create_connection = socket.create_connection
    getaddrinfo = socket.getaddrinfo
    gethostbyname = socket.gethostbyname

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        _reject(address)
        return connect(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any) -> Any:
        _reject(address)
        return connect_ex(sock, address)

    def guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        _reject(address)
        return create_connection(address, *args, **kwargs)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        _reject_name(host)
        return getaddrinfo(host, *args, **kwargs)

    def guarded_gethostbyname(host: Any) -> Any:
        _reject_name(host)
        return gethostbyname(host)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", guarded_gethostbyname)


# --------------------------------------------------------------------------- #
# Workspace / settings isolation
# --------------------------------------------------------------------------- #


def fake_api_key() -> str:
    """A syntactically plausible, obviously fake credential for offline tests."""
    return "offline-test-key-not-a-real-credential"


def stub_request() -> HttpxRequest:
    """A request object for SDK error types; never sent anywhere."""
    return HttpxRequest("POST", "https://stub.invalid")


def stub_response(status_code: int) -> HttpxResponse:
    """A response object for SDK error types; never sent anywhere."""
    return HttpxResponse(status_code, request=stub_request())


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """An empty directory the test may write into (unique per test)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@functools.lru_cache(maxsize=1)
def _scratch_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="furrow-test-scratch-"))


@pytest.fixture(scope="session")
def scratch_workspace() -> Path:
    """A shared throwaway directory outside the repo, for settings-free stubs."""
    return _scratch_dir()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every ``FURROW_*`` variable and provider key from the environment."""
    for name in list(os.environ):
        if name.startswith("FURROW_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


SettingsFactory = Callable[..., Settings]
StubFactory = Callable[..., "StubLLMClient"]
TaskFactory = Callable[..., TaskModel]
PlanFactory = Callable[..., Plan]
Responder = Callable[[str], str]


def offline_settings(**values: Any) -> Settings:
    """Build a :class:`Settings` that cannot pick up an ambient ``.env`` file.

    ``Settings`` reads ``.env`` from the current directory by default; disabling
    it (``_env_file=None``) keeps every test deterministic no matter which
    directory pytest was started from. pydantic hides the underscore-prefixed
    init arguments from its generated signature, hence the one type-ignore.
    """
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def settings_factory(tmp_path: Path) -> SettingsFactory:
    """Build a :class:`Settings` rooted in ``tmp_path``; kwargs override anything.

    ``settings_factory()`` gives cheap, test-friendly values (fake API keys,
    ``state_file`` under the temp workspace, short timeouts). Pass
    ``workspace=other_dir`` to relocate, ``state_file=None`` to fall back to
    ``<workspace>/.furrow/state.json``, or any other ``Settings`` field as a
    keyword argument. A stray repo ``.env`` can never leak in: the factory
    disables pydantic-settings' env-file lookup.
    """

    def factory(workspace: Path | None = None, **overrides: Any) -> Settings:
        root = Path(workspace) if workspace is not None else tmp_path
        values: dict[str, Any] = {
            "workspace": root,
            "provider": Provider.ANTHROPIC,
            "anthropic_api_key": fake_api_key(),
            "openai_api_key": fake_api_key(),
            "ollama_base_url": "http://127.0.0.1:11434",
            "max_tokens": 256,
            "max_cycles": 3,
            "max_parallel_tasks": 2,
            "max_consecutive_failures": 2,
            "max_task_attempts": 2,
            "test_timeout": 5,
        }
        values.setdefault("state_file", root / ".furrow" / "state.json")
        values.update(overrides)
        return offline_settings(**values)

    return factory


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #


def make_task(
    id: str = "t1",
    description: str = "do the thing",
    *,
    files: Sequence[str] | None = None,
    dependencies: Sequence[str] | None = None,
    status: TaskStatus | str = TaskStatus.PENDING,
    result: str | None = None,
    error: str | None = None,
    attempts: int = 0,
) -> TaskModel:
    """A :class:`TaskModel` with every field optional."""
    return TaskModel(
        id=id,
        description=description,
        files=list(files or []),
        dependencies=list(dependencies or []),
        status=TaskStatus(status),
        result=result,
        error=error,
        attempts=attempts,
    )


def make_plan(
    tasks: Sequence[TaskModel] | None = None,
    rationale: str = "",
) -> Plan:
    """A :class:`Plan` from ``tasks`` (empty plan by default)."""
    return Plan(tasks=list(tasks or []), rationale=rationale)


def plan_json(
    tasks: Sequence[TaskModel] | None = None,
    rationale: str = "",
) -> str:
    """The markdown-fenced JSON a planner would return for ``tasks``."""
    payload = make_plan(tasks, rationale).model_dump(mode="json")
    return f"```json\n{json.dumps(payload)}\n```"


def verdict_json(
    *,
    passed: bool = True,
    summary: str = "all checks green",
    failures: Sequence[str] = (),
) -> str:
    """The bare JSON verdict a tester would return."""
    payload = TestResult(passed=passed, summary=summary, failures=list(failures)).model_dump(
        mode="json"
    )
    return json.dumps(payload)


@pytest.fixture
def task_factory() -> TaskFactory:
    """Fixture form of :func:`make_task`."""
    return make_task


@pytest.fixture
def plan_factory() -> PlanFactory:
    """Fixture form of :func:`make_plan`."""
    return make_plan


# --------------------------------------------------------------------------- #
# LLM stub
# --------------------------------------------------------------------------- #

StubResponse = str | BaseException
StubResponses = str | Sequence[StubResponse] | None


@dataclass(frozen=True, slots=True)
class LLMCall:
    """One recorded :meth:`LLMClient.complete` call."""

    prompt: str
    system: str
    model: str | None


class StubLLMClient(LLMClient):
    """An offline :class:`~furrow.llm.LLMClient` that replays scripted replies.

    ``responses`` is a single string, or a sequence of strings / exceptions
    consumed one per call (an exception item is raised instead of returned,
    which is how a failing worker call is simulated). Once the queue is empty
    the ``responder`` callable is used if one was given, otherwise ``default``
    is returned. Every call is appended to :attr:`calls`; the prompts alone are
    available as :attr:`prompts`.

    Subclasses ``LLMClient``, so it satisfies the ``client`` parameter of the
    planner, worker, tester and orchestrator, and the real ``read_file`` /
    ``write_file`` / ``list_files`` helpers stay available.
    """

    def __init__(
        self,
        responses: StubResponses = None,
        *,
        default: str = "",
        responder: Responder | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings=settings if settings is not None else _standalone_settings())
        self.calls: list[LLMCall] = []
        self._queue: list[StubResponse] = _normalise_responses(responses)
        self._default = default
        self._responder = responder

    @property
    def prompts(self) -> list[str]:
        """Every prompt handed to :meth:`complete`, in order."""
        return [call.prompt for call in self.calls]

    @property
    def models(self) -> list[str | None]:
        """The model requested by every call, in order (``None`` means default)."""
        return [call.model for call in self.calls]

    @property
    def remaining(self) -> int:
        """How many scripted replies are still queued."""
        return len(self._queue)

    def queue(self, *responses: StubResponse) -> None:
        """Append more scripted replies to be used by later calls."""
        self._queue.extend(responses)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.calls.append(LLMCall(prompt=prompt, system=system, model=model))
        item = self._queue.pop(0) if self._queue else None
        if item is None:
            return self._responder(prompt) if self._responder is not None else self._default
        if isinstance(item, BaseException):
            raise item
        return item


def _normalise_responses(responses: StubResponses) -> list[StubResponse]:
    if responses is None:
        return []
    if isinstance(responses, str):
        return [responses]
    return list(responses)


def _standalone_settings() -> Settings:
    return offline_settings(
        workspace=_scratch_dir(),
        anthropic_api_key=fake_api_key(),
        openai_api_key=fake_api_key(),
        state_file=_scratch_dir() / ".furrow" / "state.json",
    )


@pytest.fixture
def stub_llm(settings_factory: SettingsFactory) -> StubLLMClient:
    """A stub client that answers every call with an empty string."""
    return StubLLMClient(settings=settings_factory())


@pytest.fixture
def stub_llm_factory(settings_factory: SettingsFactory) -> StubFactory:
    """Build scripted stub clients: ``make(responses, default=..., responder=...)``.

    ``settings`` defaults to a fresh :func:`settings_factory` ``Settings`` so a
    stub never points at the repository.
    """

    def factory(
        responses: StubResponses = None,
        *,
        default: str = "",
        responder: Responder | None = None,
        settings: Settings | None = None,
    ) -> StubLLMClient:
        return StubLLMClient(
            responses,
            default=default,
            responder=responder,
            settings=settings if settings is not None else settings_factory(),
        )

    return factory


@contextmanager
def no_retry_sleep() -> Iterator[None]:
    """Remove tenacity's backoff sleeps for the duration of the block.

    The retry policy hangs off the public ``LLMClient.complete`` wrapper, so
    this only affects sleeping -- attempt counts and the retry predicate are
    untouched. Used by the retry tests in ``tests/test_llm.py`` to keep them
    fast and deterministic.
    """
    policy: Any = LLMClient.complete.retry  # type: ignore[attr-defined]
    original = policy.sleep

    async def instant(_seconds: float) -> None:
        return None

    policy.sleep = instant
    try:
        yield
    finally:
        policy.sleep = original
