import json

import furrow.config
from furrow.config import Provider, Settings
from furrow.llm import LLMClient


class FakeAnthropic:
    def __init__(self, response):
        self.response = response
        self.calls = []

    @property
    def messages(self):
        return self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAI:
    def __init__(self, response):
        self.response = response
        self.calls = []

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make_anthropic_resp(text="", tool_use=None):
    class Block:
        pass

    blocks = []
    if text:
        b = Block()
        b.text = text
        b.type = "text"
        blocks.append(b)
    if tool_use:
        b = Block()
        b.type = "tool_use"
        b.id = tool_use["id"]
        b.name = tool_use["name"]
        b.input = tool_use.get("input", {})
        blocks.append(b)
    resp = Block()
    resp.content = blocks
    resp.stop_reason = "stop"
    return resp


def make_openai_resp(content=None, tool_calls=None):
    class Msg:
        pass

    m = Msg()
    m.content = content
    m.tool_calls = tool_calls

    class Choice:
        pass

    c = Choice()
    c.message = m
    c.finish_reason = "stop"

    class Resp:
        pass

    r = Resp()
    r.choices = [c]
    return r


def make_openai_tool_call(id, name, args):
    class Fn:
        pass

    f = Fn()
    f.name = name
    f.arguments = json.dumps(args)

    class TC:
        pass

    t = TC()
    t.id = id
    t.function = f
    return t


async def test_singleton():
    client = LLMClient()
    assert client.settings is furrow.config.settings


async def test_provider_routing(tmp_path):
    for provider, fake_attr in [
        (Provider.ANTHROPIC, "_anthropic"),
        (Provider.OPENAI, "_openai"),
        (Provider.OLLAMA, "_ollama"),
    ]:
        settings = Settings(provider=provider)
        client = LLMClient(settings)
        if provider == Provider.ANTHROPIC:
            fake = FakeAnthropic(make_anthropic_resp(text="hi"))
        else:
            fake = FakeOpenAI(make_openai_resp(content="hi"))
        setattr(client, fake_attr, fake)
        out = await client.complete("prompt")
        assert out == "hi"
        assert fake.calls


async def test_anthropic_mixed_content():
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings)

    class NoText:
        type = "other"

    class TextBlock:
        type = "text"
        text = "hello"

    resp = type("R", (), {})()
    resp.content = [TextBlock(), NoText()]
    resp.stop_reason = "stop"
    client._anthropic = FakeAnthropic(resp)
    out = await client.complete("p")
    assert out == "hello"


async def test_openai_none_content():
    settings = Settings(provider=Provider.OPENAI)
    client = LLMClient(settings)
    client._openai = FakeOpenAI(make_openai_resp(content=None))
    out = await client.complete("p")
    assert out == ""


async def test_read_write_roundtrip(tmp_path):
    p = tmp_path / "sub" / "file.txt"
    client = LLMClient()
    await client.write_file(p, "héllo — ✓")
    got = await client.read_file(p)
    assert got == "héllo — ✓"
    assert p.parent.is_dir()


async def test_list_files(tmp_path):
    (tmp_path / "main.py").write_text("x")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x")
    (tmp_path / ".env").write_text("x")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_text("x")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "a.js").write_text("x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("x")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.txt").write_text("x")
    client = LLMClient()
    res = client.list_files(tmp_path)
    assert "main.py" in res
    assert "pkg/mod.py" in res
    for bad in [".env", "__pycache__/x.pyc", "node_modules/a.js", ".git/config", "build/out.txt"]:
        assert bad not in res
    assert client.list_files(tmp_path / "nope") == []
    client.list_files(tmp_path, ignore=["secret.txt"])
    (tmp_path / "secret.txt").write_text("x")
    # ignore applied at call time already; re-check with file present
    res3 = client.list_files(tmp_path, ignore=["secret.txt"])
    assert "secret.txt" not in res3


async def test_complete_with_tools_anthropic():
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings)
    resp = make_anthropic_resp(text="hi", tool_use={"id": "1", "name": "foo", "input": {"a": 1}})
    client._anthropic = FakeAnthropic(resp)
    r = await client.complete_with_tools("p", tools=[{"name": "foo", "input_schema": {}}])
    assert r.tool_calls[0].name == "foo"
    assert r.tool_calls[0].arguments == {"a": 1}
    assert r.text == "hi"


async def test_run_tool_loop_anthropic():
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings)

    first = make_anthropic_resp(text="", tool_use={"id": "t1", "name": "foo", "input": {"x": 2}})
    second = make_anthropic_resp(text="done")
    fake = FakeAnthropic(first)
    # make create return first then second

    async def create(**kwargs):
        fake.calls.append(kwargs)
        if len(fake.calls) == 1:
            return first
        return second

    fake.create = create
    client._anthropic = fake

    executed = []

    async def execute(tc):
        executed.append(tc)
        return "result"

    out = await client.run_tool_loop("p", execute)
    assert len(executed) == 1
    assert executed[0].name == "foo"
    assert executed[0].arguments == {"x": 2}
    assert out == "done"
    assert len(fake.calls) == 2
    # second request messages contains tool result
    second_call_msgs = fake.calls[1]["messages"]
    assert any(
        (
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        for m in second_call_msgs
        if isinstance(m, dict)
    )


async def test_run_tool_loop_max_rounds():
    settings = Settings(provider=Provider.ANTHROPIC)
    client = LLMClient(settings)
    tool_resp = make_anthropic_resp(text="", tool_use={"id": "t1", "name": "foo", "input": {}})

    class Fake:
        def __init__(self):
            self.calls = []

        @property
        def messages(self):
            return self

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return tool_resp

    fake = Fake()
    client._anthropic = fake

    async def execute(tc):
        return "r"

    await client.run_tool_loop("p", execute, max_rounds=2)
    assert len(fake.calls) <= 2
