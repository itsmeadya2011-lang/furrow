"""Quick checks for ``get_command_denied_reason``.

Each denied command must produce a non-empty human-readable reason, each
allowed command must produce an empty string, and the reason must agree with
``is_command_denied`` for every input.

Run directly with ``python3 tests/test_denied_reason.py`` or via pytest.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from furrow.agents.worker import get_command_denied_reason, is_command_denied

# (command, expected reason substring). The substring is checked case-sensitively
# so a matching reason points at the right policy rule.
DENIED_CASES = [
    ("rm -rf /", "dangerous rm"),
    ("rm -r -f ~", "dangerous rm"),
    ("mkfs.ext4 /dev/sda", "mkfs"),
    ("dd if=/dev/zero of=/dev/sda", "dd if="),
    ("fdisk /dev/sda", "fdisk"),
    ("shutdown -h now", "shutdown"),
    ("reboot", "reboot"),
    ("parted /dev/sda", "parted"),
    (":(){:|:&};:", ":(){:|:&};:"),
    # wrapper programs resolve to the denied program
    ("sudo fdisk /dev/sda", "fdisk"),
    ("env FOO=1 shutdown -h now", "shutdown"),
    # force push
    ("git push --force", "force push"),
    ("git push origin +main", "force push"),
    # writes outside the workspace
    ("echo hacked > /etc/passwd", "redirection outside workspace"),
    ("echo hi | tee /etc/passwd", "write outside workspace"),
    # dangerous chmod
    ("chmod -R 777 /", "chmod"),
    # substitutions / process substitutions
    ("echo $(fdisk /dev/sda)", "fdisk"),
    ("cat <(mkfs.ext4 /dev/sda)", "mkfs"),
]

ALLOWED_CASES = [
    "pytest -q",
    "git status",
    "echo hi > out.txt",
    "rm -rf build",
    "ls -f",
    "",
]


def _check(workspace: Path) -> list[str]:
    failures: list[str] = []

    for command, expected in DENIED_CASES:
        reason = get_command_denied_reason(command, workspace)
        if not reason:
            failures.append(f"denied but no reason: {command!r}")
        elif expected not in reason:
            failures.append(
                f"unexpected reason for {command!r}: {reason!r} (expected to contain {expected!r})"
            )
        if not is_command_denied(command, workspace):
            failures.append(
                "get_command_denied_reason says denied but is_command_denied says allowed: "
                f"{command!r}"
            )

    for command in ALLOWED_CASES:
        reason = get_command_denied_reason(command, workspace)
        if reason:
            failures.append(f"allowed command got a reason: {command!r} -> {reason!r}")
        if is_command_denied(command, workspace):
            failures.append(f"is_command_denied denies an allowed command: {command!r}")

    return failures


def test_get_command_denied_reason(tmp_path: Path) -> None:
    failures = _check(tmp_path)
    assert not failures, "\n".join(failures)


def _main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        failures = _check(Path(tmp))
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(f"OK: {len(DENIED_CASES)} denied and {len(ALLOWED_CASES)} allowed cases verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())