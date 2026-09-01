from __future__ import annotations

import pytest

from furrow.config import Plan, Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_orchestrator_init_has_latest_plan_attr() -> None:
    orchestrator = Orchestrator(goal="x")
    assert orchestrator._latest_plan is None
    assert orchestrator._get_tasks() == []


@pytest.mark.asyncio
async def test_orchestrator_accepts_output_callback() -> None:
    async def cb(msg: str) -> None:
        pass

    orchestrator = Orchestrator(goal="x", output_callback=cb)
    assert orchestrator.output_callback is cb


@pytest.mark.asyncio
async def test_orchestrator_emit_swallows_exceptions() -> None:
    async def cb(msg: str) -> None:
        raise RuntimeError("boom")

    orchestrator = Orchestrator(goal="x", output_callback=cb)
    await orchestrator._emit("hello")


def test_provider_enum_has_ollama() -> None:
    assert Provider.OLLAMA.value == "ollama"


def test_settings_max_cycles_default() -> None:
    assert Settings().max_cycles == 0


def test_llm_client_ollama_property_exists() -> None:
    c = LLMClient()
    assert hasattr(c, "ollama")


def test_plan_model_with_no_tasks() -> None:
    p = Plan(tasks=[], rationale="empty")
    assert p.tasks == []
