"""Command-line entry point for Furrow.

The commands here only translate CLI flags into the global :data:`furrow.config.settings`
object that the orchestrator and the agents read from, then hand off. Two rules keep that
translation honest:

* every agent call passes an explicit role model, so ``--model`` must set *all* of them --
  setting ``settings.model`` alone has no effect;
* the built-in role models are Anthropic ids, so running against another provider would
  send Anthropic model names to that provider's API. :data:`PROVIDER_DEFAULT_MODELS` fills
  in per-provider defaults for any model the user did not configure explicitly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from furrow.config import ENV_PREFIX, Provider, Settings, settings

try:  # `configure_logging` lives in furrow.config; stay importable if it ever moves.
    from furrow.config import configure_logging
except ImportError:  # pragma: no cover - helper not yet available

    def configure_logging(level: str | None = None) -> None:
        logging.basicConfig(level=(level or "INFO").upper())


console = Console()
logger = logging.getLogger(__name__)

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

#: `Settings` fields that hold a model id, keyed exactly like :data:`PROVIDER_DEFAULT_MODELS`.
MODEL_FIELDS = ("model", "planner_model", "worker_model", "tester_model")

#: Default model per provider, for both `settings.model` and the three role models.
#: The anthropic entry mirrors the defaults declared in `furrow.config`; the others keep a
#: non-anthropic provider from receiving Anthropic model ids (which answer 404). Any field
#: the user set explicitly (env, `.env`, or `--model`) is left alone.
PROVIDER_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "anthropic": {
        "model": "claude-sonnet-4-20250514",
        "planner_model": "claude-3-5-haiku-20241022",
        "worker_model": "claude-3-5-sonnet-20241022",
        "tester_model": "claude-3-5-sonnet-20241022",
    },
    "openai": {
        "model": "gpt-4o",
        "planner_model": "gpt-4o-mini",
        "worker_model": "gpt-4o",
        "tester_model": "gpt-4o",
    },
    "ollama": {
        "model": "qwen2.5-coder:7b",
        "planner_model": "qwen2.5-coder:7b",
        "worker_model": "qwen2.5-coder:7b",
        "tester_model": "qwen2.5-coder:7b",
    },
}

#: Imported by `uvicorn[standard]` / fastapi; used to give a targeted install hint.
_WEB_DEPENDENCIES = frozenset(
    {"fastapi", "uvicorn", "starlette", "websockets", "httptools", "uvloop", "watchfiles"}
)


def _package_version() -> str:
    """Distribution version for `--version`, falling back outside an installed env."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("furrow")
    except PackageNotFoundError:  # pragma: no cover - running from a source checkout
        return "0.0.0"


def _provider_choices() -> tuple[str, ...]:
    """Provider values accepted by `--provider`, derived from the `Provider` enum."""
    return tuple(member.value for member in Provider)


def _provider_value(provider: Any) -> str:
    """`Provider` enum member or plain string -> lowercase value."""
    return str(getattr(provider, "value", provider)).lower()


def _declared_default(field_name: str) -> Any:
    """Default declared on the `Settings` model field, or None when there is none."""
    fields = getattr(Settings, "model_fields", None) or getattr(Settings, "__fields__", {})
    return getattr(fields.get(field_name), "default", None)


def _model_is_configured(field_name: str) -> bool:
    """Whether `field_name` holds a user-supplied model rather than a built-in default.

    Env vars win explicitly (Furrow env names are case-insensitive); otherwise the value is
    compared against the model default, which also covers values coming from a `.env` file.
    """
    env_name = f"furrow_{field_name}".lower()
    if any(key.lower() == env_name for key in os.environ):
        return True
    default = _declared_default(field_name)
    if default is None:
        return False
    return getattr(settings, field_name, None) != default


def _apply_provider_defaults(provider: str) -> None:
    """Point unset model fields at `provider`'s defaults (see PROVIDER_DEFAULT_MODELS)."""
    defaults = PROVIDER_DEFAULT_MODELS.get(provider.lower())
    if not defaults:
        return
    for field_name, model in defaults.items():
        if not _model_is_configured(field_name):
            setattr(settings, field_name, model)


def _resolve_log_level(log_level: str | None, verbose: bool) -> str:
    """`--log-level` wins, then `-v`, then the configured `FURROW_LOG_LEVEL`, else INFO."""
    if log_level:
        return log_level.upper()
    if verbose:
        return "DEBUG"
    return str(getattr(settings, "log_level", None) or "INFO").upper()


def _runtime_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Options shared by every command that starts a Furrow run."""
    options = (
        click.option(
            "--provider",
            type=click.Choice(_provider_choices()),
            default=None,
            help="LLM provider (default: $FURROW_PROVIDER, else anthropic).",
        ),
        click.option(
            "--model",
            default=None,
            metavar="MODEL",
            help="Model for every role (planner, worker, tester).",
        ),
        click.option(
            "--workspace",
            type=click.Path(exists=True, file_okay=False, path_type=Path),
            default=None,
            help="Directory to work in (default: $FURROW_WORKSPACE, else cwd).",
        ),
        click.option(
            "--log-level",
            type=click.Choice(LOG_LEVELS, case_sensitive=False),
            default=None,
            metavar="LEVEL",
            help=f"Log level: {', '.join(LOG_LEVELS)} (default: $FURROW_LOG_LEVEL, else INFO).",
        ),
        click.option("-v", "--verbose", is_flag=True, help="Shorthand for --log-level DEBUG."),
    )
    for option in options:
        func = option(func)
    return func


def _apply_runtime_options(
    *,
    provider: str | None,
    model: str | None,
    workspace: Path | None,
    log_level: str | None,
    verbose: bool,
) -> None:
    """Fold CLI flags into the global settings object the orchestrator and agents read."""
    if provider:
        settings.provider = Provider(provider)
    if model:
        # Agents pass an explicit role model on every call, so `settings.model` is never
        # reached on its own: --model has to win for all four fields to take effect.
        for field_name in MODEL_FIELDS:
            setattr(settings, field_name, model)
    else:
        # Covers both `--provider openai` and FURROW_PROVIDER=openai from the environment.
        _apply_provider_defaults(_provider_value(settings.provider))

    if workspace is not None:
        settings.workspace = workspace
    settings.log_level = _resolve_log_level(log_level, verbose)
    configure_logging(settings.log_level)
    logger.debug(
        "provider=%s planner=%s worker=%s tester=%s workspace=%s max_cycles=%s log_level=%s",
        settings.provider,
        settings.planner_model,
        settings.worker_model,
        settings.tester_model,
        settings.workspace,
        settings.max_cycles,
        settings.log_level,
    )


#: Where each provider's API key may come from: provider value -> (Settings field, bare env
#: var). Mirrors the lazy lookup in `LLMClient.anthropic` / `.openai` (`furrow/llm.py`):
#: the `FURROW_`-prefixed setting (environment or `.env` file) first, the bare provider
#: environment variable as fallback -- either source alone is enough.
_CREDENTIAL_SOURCES: dict[str, tuple[str, str]] = {
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
}


def _require_credentials(resolved: Settings | None = None) -> None:
    """Refuse to start a run when the effective provider has no API key.

    `LLMClient` resolves its key lazily, so a missing one otherwise surfaces as a
    `ValueError` inside the first agent call -- where the orchestrator's per-cycle
    `except Exception` swallows it and the command still exits 0 with tracebacks on
    stderr. Checking here turns it into one clean error and a non-zero exit.
    """
    config = resolved if resolved is not None else settings
    source = _CREDENTIAL_SOURCES.get(_provider_value(config.provider))
    if source is None:
        # Ollama needs no key, and an unknown provider is reported by `LLMClient` itself.
        return
    field_name, env_var = source
    if getattr(config, field_name, None) or os.getenv(env_var):
        return
    raise click.ClickException(
        f"{env_var} is not set (provider {_provider_value(config.provider)}). "
        f"Export {env_var} in your environment, or set it in a .env file as "
        f"{ENV_PREFIX}{field_name.upper()}."
    )


def _dependency_error(exc: ImportError) -> click.ClickException:
    """Turn a missing optional dependency into an actionable message."""
    module = exc.name or "a required package"
    if module.split(".")[0] in _WEB_DEPENDENCIES:
        hint = 'pip install "uvicorn[standard]" fastapi'
    else:
        hint = "pip install -e .  # reinstall Furrow's dependencies"
    return click.ClickException(f"Missing dependency '{module}'. Install it with: {hint}")


@contextmanager
def _user_errors() -> Iterator[None]:
    """Report expected failure modes as CLI errors instead of tracebacks.

    Only `ValueError` (missing API key, unsupported provider, ...), `KeyboardInterrupt` and
    missing-dependency `ImportError` are handled; anything else propagates so real bugs are
    never hidden.
    """
    try:
        yield
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except ImportError as exc:
        raise _dependency_error(exc) from exc
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _no_goal_error() -> click.ClickException:
    """The single "no usable goal" error, always saying how to supply one.

    Both ways the prompt can come up empty -- Ctrl-D/EOF on an empty stdin and a bare
    Enter -- get this same actionable message rather than a bare "No goal provided.".
    """
    return click.ClickException(
        'No goal provided. Pass it as an argument: furrow start "add a /health endpoint"'
    )


def _prompt_for_goal() -> str:
    """Ask for the goal when none was given; abort cleanly if nothing usable comes back."""
    try:
        goal = click.prompt("Enter your goal for Furrow", default="", show_default=False)
    except (click.Abort, EOFError):
        raise _no_goal_error() from None
    goal = goal.strip()
    if not goal:
        raise _no_goal_error()
    return goal


@click.group()
@click.version_option(version=_package_version(), prog_name="furrow")
def main() -> None:
    """Furrow: an autonomous coding agent that loops plan -> execute -> test."""


@main.command()
@click.argument("goal", required=False)
@_runtime_options
@click.option(
    "--max-cycles",
    type=click.IntRange(min=0),
    default=None,
    metavar="N",
    help="Stop after N cycles (0 = unlimited; default: $FURROW_MAX_CYCLES).",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    workspace: Path | None,
    log_level: str | None,
    verbose: bool,
    max_cycles: int | None,
) -> None:
    """Run the Furrow loop until the goal is met or a limit is reached."""
    if not goal or not goal.strip():
        goal = _prompt_for_goal()

    _apply_runtime_options(
        provider=provider,
        model=model,
        workspace=workspace,
        log_level=log_level,
        verbose=verbose,
    )
    if max_cycles is not None:
        settings.max_cycles = max_cycles

    with _user_errors():
        # Fail before the first (patchable) agent call: see `_require_credentials`.
        _require_credentials()

        # Lazy import: keeps `furrow start` usable without the web extras installed.
        from furrow.core.orchestrator import Orchestrator

        asyncio.run(Orchestrator(goal=goal, settings=settings).run())


@main.command()
@_runtime_options
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to bind (use 0.0.0.0 to accept remote connections).",
)
@click.option(
    "--port",
    type=click.IntRange(0, 65535),
    default=8000,
    show_default=True,
    help="Port to bind.",
)
@click.option("--reload", is_flag=True, help="Restart the server when the code changes.")
def web(
    host: str,
    port: int,
    reload: bool,
    model: str | None,
    provider: str | None,
    workspace: Path | None,
    log_level: str | None,
    verbose: bool,
) -> None:
    """Serve the Furrow web UI."""
    _apply_runtime_options(
        provider=provider,
        model=model,
        workspace=workspace,
        log_level=log_level,
        verbose=verbose,
    )
    with _user_errors():
        # Lazy import: fastapi/uvicorn are only needed here, not by `furrow start`.
        from furrow.web.server import run

        run(host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
