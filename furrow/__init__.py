__all__ = ["LLMClient", "Settings"]


def __getattr__(name: str):
    if name == "LLMClient":
        from furrow.llm import LLMClient
        return LLMClient
    if name == "Settings":
        from furrow.config import Settings
        return Settings
    raise AttributeError(f"module 'furrow' has no attribute '{name}'")