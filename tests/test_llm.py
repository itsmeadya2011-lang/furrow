import json

from furrow.config import Settings
from furrow.llm import LLMClient, LLMResponse, Provider, Tool, ToolCall


class _Func:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _OpenAItc:
    def __init__(self, id, name, arguments_str):
        self.id = id
        self.type = "function"
        self.function = _Func(name, arguments_str)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message):
        self.message = message


class _Resp:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeChat:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)


class _FakeOpenAI:
    def __init__(self, responses):
        self.chat = _FakeChat(responses)


def _client(provider=Provider.OPENAI, **overrides):
    kwargs = {"provider": provider, "openai_api_key": "fake"}
    kwargs.update(overrides)
    return LLMClient(settings=Settings(**kwargs))


async def test_tool_dataclasses():
    t = Tool(name="n", description="d", input_schema={"type": "object"})
    assert t.name == "n"
    tc = ToolCall(id="1", name="n", arguments={"a": 1})
    assert tc.arguments == {"a": 1}
    r = LLMResponse(text="hi", tool_calls=[tc])
    assert r.text == "hi"
    assert r.tool_calls[0].id == "1"


async def test_chat_openai_text_only():
    client = _client()
    fake = _FakeOpenAI([_Resp(_Msg(content="done", tool_calls=None))])
    client._openai = fake
    resp = await client.chat(
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
        system="",
        model="x",
    )
    assert resp.text == "done"
    assert resp.tool_calls == []


async def test_chat_openai_parses_tool_call():
    client = _client()
    args = json.dumps({"path": "x"})
    fake = _FakeOpenAI([
        _Resp(_Msg(content=None, tool_calls=[_OpenAItc("call_1", "read_file", args)]))
    ])
    client._openai = fake
    tool = Tool(name="read_file", description="r", input_schema={"type": "object"})
    resp = await client.chat(
        tools=[tool],
        messages=[{"role": "user", "content": "hi"}],
        system="sys",
        model="x",
    )
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "x"}
    call = fake.chat.completions.calls[-1]
    assert len(call["messages"]) == 2
    assert call["tools"][0]["function"]["name"] == "read_file"


async def test_chat_ollama_uses_fake_client(monkeypatch):
    client = _client(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    fake_client = _FakeOpenAI([_Resp(_Msg(content="ollama-done", tool_calls=None))])
    monkeypatch.setattr(client, "_ollama_client", lambda: fake_client)
    resp = await client.chat(
        tools=[],
        messages=[{"role": "user", "content": "hi"}],
        system="",
        model="x",
    )
    assert resp.text == "ollama-done"


async def test_ollama_client_url():
    client = _client(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    c = client._ollama_client()
    assert "11434" in str(c.base_url)
