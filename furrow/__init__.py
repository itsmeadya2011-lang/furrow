from furrow.config import Provider, Settings, TaskModel, Plan, TestResult

__all__ = [
    "Provider",
    "Settings",
    "TaskModel",
    "Plan",
    "TestResult",
    "LLMClient",
    "NonRetryableError",
]


def __getattr__(name: str):
    if name in ("LLMClient", "NonRetryableError"):
        from furrow.llm import LLMClient, NonRetryableError
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
