from __future__ import annotations

import asyncio

import pytest

from furrow.git import Git, GitResult, _is_force_push, _is_safe, git_result_from_exception


class FakeProc:
    """Stand-in for an asyncio subprocess with fixed output and exit status."""

    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.pid = 4242
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr

    async def wait(self):
        return self.returncode

    def kill(self):
        self.returncode = -9


def run(coro):
    """Run a coroutine to completion on a fresh event loop."""
    return asyncio.run(coro)


def fake_subprocess(returncode=0, stdout=b"", stderr=b""):
    """Return a fake create_subprocess_exec that yields a FakeProc."""

    async def factory(*args, **kwargs):
        return FakeProc(returncode=returncode, stdout=stdout, stderr=stderr)

    return factory


# -- GitResult ---------------------------------------------------------------


def test_git_result_defaults():
    r = GitResult()
    assert r.ok is False
    assert r.stdout == ""
    assert r.stderr == ""
    assert r.command == ""
    assert r.error is None


def test_git_result_combined():
    r = GitResult(ok=True, stdout="ok out", stderr="err out", command="git status")
    assert "[stderr]" in r.combined
    assert "ok out" in r.combined
    assert "err out" in r.combined


def test_git_result_combined_no_stderr():
    r = GitResult(ok=True, stdout="just stdout", stderr="", command="git status")
    assert r.combined == "just stdout"


# -- Safety helpers ----------------------------------------------------------


def test_is_force_push_detects_flags():
    assert _is_force_push(["git", "push", "--force"]) is True
    assert _is_force_push(["git", "push", "--force-with-lease"]) is True
    assert _is_force_push(["git", "push", "-f", "origin", "main"]) is True
    assert _is_force_push(["git", "push", "origin", "main"]) is False
    assert _is_force_push(["git", "commit", "-m", "x"]) is False
    assert _is_force_push(["ls", "-f"]) is False


def test_is_safe_rejects_force_push():
    assert _is_safe(["git", "push", "--force"]) is False
    assert _is_safe(["git", "push", "-f"]) is False


def test_is_safe_rejects_destructive_helpers():
    assert _is_safe(["git", "config", "core.sshCommand", "dd if=/dev/zero"]) is False
    assert _is_safe(["git", "config", "core.sshCommand", "mkfs.ext4 /dev/sda"]) is False
    assert _is_safe(["git", "status"]) is True


def test_is_safe_allows_ordinary_git():
    assert _is_safe(["git", "status", "--porcelain"]) is True
    assert _is_safe(["git", "add", "-A"]) is True
    assert _is_safe(["git", "commit", "-m", "msg"]) is True
    assert _is_safe(["git", "push", "origin", "main"]) is True
    assert _is_safe(["git", "rev-parse", "--abbrev-ref", "HEAD"]) is True


# -- Git: repo detection -----------------------------------------------------


def test_is_repo_true(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=0, stdout=b"true\n"))
    git = Git(workspace=tmp_path)
    assert run(git.is_repo()) is True


def test_is_repo_false(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=1, stdout=b"false\n"))
    git = Git(workspace=tmp_path)
    assert run(git.is_repo()) is False


# -- Git: current branch -----------------------------------------------------


def test_current_branch(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=0, stdout=b"main\n"))
    git = Git(workspace=tmp_path)
    assert run(git.current_branch()) == "main"


def test_current_branch_empty_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=1, stdout=b""))
    git = Git(workspace=tmp_path)
    assert run(git.current_branch()) == ""


# -- Git: status -------------------------------------------------------------


def test_status_porcelain(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=0, stdout=b" M foo.py\n?? bar.py\n"))
    git = Git(workspace=tmp_path)
    result = run(git.status())
    assert result.ok is True
    assert "foo.py" in result.stdout
    assert "bar.py" in result.stdout


# -- Git: is_dirty -----------------------------------------------------------


def test_is_dirty_true(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=0, stdout=b" M foo.py\n"))
    git = Git(workspace=tmp_path)
    assert run(git.is_dirty()) is True


def test_is_dirty_false(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=0, stdout=b""))
    git = Git(workspace=tmp_path)
    assert run(git.is_dirty()) is False


# -- Git: remote -------------------------------------------------------------


def test_remote_url(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=0, stdout=b"git@github.com:user/repo.git\n"))
    git = Git(workspace=tmp_path)
    url = run(git.remote_url())
    assert url == "git@github.com:user/repo.git"


def test_remote_url_none(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=1, stdout=b""))
    git = Git(workspace=tmp_path)
    assert run(git.remote_url()) is None


def test_has_remote(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=0, stdout=b"git@github.com:user/repo.git\n"))
    git = Git(workspace=tmp_path)
    assert run(git.has_remote()) is True


# -- Git: add ----------------------------------------------------------------


def test_add_all(monkeypatch, tmp_path):
    captured = {}

    async def factory(*args, **kwargs):
        captured["args"] = args
        return FakeProc(returncode=0, stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    git = Git(workspace=tmp_path)
    result = run(git.add())
    assert result.ok is True
    assert captured["args"] == ["git", "add", "-A"]


def test_add_specific_paths(monkeypatch, tmp_path):
    captured = {}

    async def factory(*args, **kwargs):
        captured["args"] = args
        return FakeProc(returncode=0, stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    git = Git(workspace=tmp_path)
    result = run(git.add(paths=["foo.py", "bar.py"]))
    assert result.ok is True
    assert captured["args"] == ["git", "add", "foo.py", "bar.py"]


# -- Git: commit -------------------------------------------------------------


def test_commit_success(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=0, stdout=b"[main abc123] msg\n"))
    git = Git(workspace=tmp_path)
    result = run(git.commit("add feature"))
    assert result.ok is True


def test_commit_nothing_to_commit_is_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=1, stdout=b"",
                                        stderr=b"nothing to commit, working tree clean\n"))
    git = Git(workspace=tmp_path)
    result = run(git.commit("msg"))
    assert result.ok is True


# -- Git: push ---------------------------------------------------------------


def test_push_with_upstream(monkeypatch, tmp_path):
    captured = {}

    async def factory(*args, **kwargs):
        captured["args"] = args
        return FakeProc(returncode=0, stdout=b"To origin\n   main -> main\n")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    git = Git(workspace=tmp_path)
    result = run(git.push(branch="main", set_upstream=True))
    assert result.ok is True
    assert captured["args"] == ["git", "push", "-u", "origin", "main"]


def test_push_without_upstream(monkeypatch, tmp_path):
    captured = {}

    async def factory(*args, **kwargs):
        captured["args"] = args
        return FakeProc(returncode=0, stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    git = Git(workspace=tmp_path)
    result = run(git.push(branch="main", set_upstream=False))
    assert result.ok is True
    assert captured["args"] == ["git", "push", "origin", "main"]


def test_push_no_branch(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=1, stdout=b""))
    git = Git(workspace=tmp_path)
    result = run(git.push(branch=""))
    assert result.ok is False
    assert "branch" in (result.error or "")


# -- Git: force push denied --------------------------------------------------


def test_push_force_denied(monkeypatch, tmp_path):
    called = {}

    async def factory(*args, **kwargs):
        called["args"] = args
        return FakeProc(returncode=0, stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    git = Git(workspace=tmp_path)
    result = run(git._run(["git", "push", "--force"]))
    assert result.ok is False
    assert "denied" in (result.error or "")


# -- Git: timeout ------------------------------------------------------------


def test_command_timeout(monkeypatch, tmp_path):
    async def factory(*args, **kwargs):
        proc = FakeProc()
        async def communicate():
            await asyncio.sleep(10)
            return b"", b""
        proc.communicate = communicate
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    git = Git(workspace=tmp_path, timeout=0.1)
    result = run(git.status())
    assert result.ok is False
    assert "timed out" in (result.error or "")


# -- Git: commit_and_push ----------------------------------------------------


def test_commit_and_push(monkeypatch, tmp_path):
    calls = []

    async def factory(*args, **kwargs):
        calls.append(list(args))
        if args[1] == "add":
            return FakeProc(returncode=0, stdout=b"")
        if args[1] == "commit":
            return FakeProc(returncode=0, stdout=b"[main abc123] msg\n")
        if args[1] == "push":
            return FakeProc(returncode=0, stdout=b"To origin\n")
        return FakeProc(returncode=0, stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    git = Git(workspace=tmp_path)
    result = run(git.commit_and_push("add feature"))
    assert result.ok is True
    assert len(calls) == 3  # add, commit, push


# -- Git: fast_forward_base --------------------------------------------------


def test_fast_forward_base(monkeypatch, tmp_path):
    calls = []

    async def factory(*args, **kwargs):
        calls.append(list(args))
        if args[1] == "checkout":
            return FakeProc(returncode=0, stdout=b"Switched to branch 'main'\n")
        if args[1] == "merge":
            return FakeProc(returncode=0, stdout=b"Already up to date.\n")
        if args[1] == "push":
            return FakeProc(returncode=0, stdout=b"To origin\n")
        return FakeProc(returncode=0, stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    git = Git(workspace=tmp_path)
    result = run(git.fast_forward_base("main"))
    assert result.ok is True
    assert any(a == ["git", "checkout", "main"] for a in calls)
    assert any(a == ["git", "merge", "--ff-only", "@{u}"] for a in calls)


# -- Git: ensure_pushed ------------------------------------------------------


def test_ensure_pushed_already_tracked(monkeypatch, tmp_path):
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        fake_subprocess(returncode=0, stdout=b"abc123\n"))
    git = Git(workspace=tmp_path)
    result = run(git.ensure_pushed(branch="main"))
    assert result.ok is True


def test_ensure_pushed_pushes(monkeypatch, tmp_path):
    calls = []

    async def factory(*args, **kwargs):
        calls.append(list(args))
        if args[1] == "rev-parse":
            return FakeProc(returncode=1, stdout=b"")
        if args[1] == "push":
            return FakeProc(returncode=0, stdout=b"To origin\n")
        return FakeProc(returncode=0, stdout=b"")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", factory)
    git = Git(workspace=tmp_path)
    result = run(git.ensure_pushed(branch="main"))
    assert result.ok is True
    assert any(a[1] == "push" for a in calls)


# -- git_result_from_exception -----------------------------------------------


def test_git_result_from_exception():
    try:
        raise OSError("disk full")
    except OSError as exc:
        r = git_result_from_exception(exc, command="git status")
        assert r.ok is False
        assert r.command == "git status"
        assert "OSError" in r.error
        assert "disk full" in r.error
