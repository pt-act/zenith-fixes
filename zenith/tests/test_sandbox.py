"""Sandbox backend tests (phase 1: bwrap argv + unshare integration)."""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from zenith_harness.sandbox import (
    BwrapSandbox,
    NoSandbox,
    SandboxUnavailable,
    UnshareSandbox,
    get_sandbox,
    prepare_node_home,
    sandbox_env_for_provider,
)


def test_no_sandbox_is_passthrough(tmp_path: Path) -> None:
    sandbox = NoSandbox()
    argv = ["/bin/echo", "hi"]
    assert sandbox.wrap(
        argv,
        cwd=tmp_path,
        env={"PATH": "/usr/bin"},
        workspace=tmp_path,
        home=tmp_path,
    ) == argv


def test_bwrap_command_shape(tmp_path: Path) -> None:
    fake = tmp_path / "bwrap"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    ws = tmp_path / "ws"
    ws.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    sandbox = BwrapSandbox(binary=str(fake))
    assert sandbox.available()
    argv = sandbox.wrap(
        ["/bin/echo", "hi"],
        cwd=ws,
        env={"PATH": "/usr/bin", "HOME": str(home)},
        workspace=ws,
        home=home,
    )
    assert argv[0] == str(fake)
    assert "--unshare-user" in argv
    assert "--die-with-parent" in argv
    assert "--clearenv" not in argv
    assert "--setenv" not in argv
    assert "--" in argv
    assert argv[-2:] == ["/bin/echo", "hi"]
    # workspace and home are bound read-write at their absolute paths
    assert ["--bind", str(ws), str(ws)] == argv[
        argv.index("--bind") : argv.index("--bind") + 3
    ]


def test_bwrap_rejects_cwd_outside_workspace(tmp_path: Path) -> None:
    fake = tmp_path / "bwrap"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    sandbox = BwrapSandbox(binary=str(fake))
    with pytest.raises(SandboxUnavailable):
        sandbox.wrap(
            ["/bin/echo"],
            cwd=tmp_path,
            env={},
            workspace=tmp_path / "ws",
            home=tmp_path / "home",
        )


def _unshare_or_skip() -> UnshareSandbox:
    sandbox = UnshareSandbox()
    if not sandbox.available():
        pytest.skip("unprivileged user namespaces/unshare/pivot_root unavailable")
    return sandbox


def test_unshare_blocks_outside_read_and_allows_workspace_write(tmp_path: Path) -> None:
    sandbox = _unshare_or_skip()
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("product\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("HOST-SECRET\n", encoding="utf-8")
    home = tmp_path / "home"
    home.mkdir()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home)}
    script = (
        f"set -e; "
        f"if cat {tmp_path}/secret.txt >/dev/null 2>&1; then echo LEAK; exit 3; fi; "
        f"echo ok > {ws}/inside.txt; "
        "test -r /etc/passwd; "
        "echo done"
    )
    argv = sandbox.wrap(
        ["/bin/sh", "-c", script],
        cwd=ws,
        env=env,
        workspace=ws,
        home=home,
    )
    result = subprocess.run(
        argv, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "done" in result.stdout
    assert (ws / "inside.txt").read_text() == "ok\n"


def test_unshare_launch_config_is_private(tmp_path: Path) -> None:
    sandbox = _unshare_or_skip()
    ws = tmp_path / "ws"
    ws.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    sandbox.wrap(
        ["/bin/true"],
        cwd=ws,
        env={"PATH": "/usr/bin", "HOME": str(home), "SECRET": "do-not-leak"},
        workspace=ws,
        home=home,
    )
    configs = list(home.parent.glob("launch-*.json"))
    assert configs
    assert stat.S_IMODE(home.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(configs[0].stat().st_mode) == 0o600


def test_prepare_node_home_is_private(tmp_path: Path) -> None:
    home = prepare_node_home("codex", tmp_path / "home")
    assert home.exists()
    assert stat.S_IMODE(home.stat().st_mode) == 0o700


def test_sandbox_env_points_at_node_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    env = sandbox_env_for_provider("codex", home, {"PATH": "/usr/bin"})
    assert env["HOME"] == str(home)
    assert env["CODEX_HOME"] == str(home / ".codex")


def test_get_sandbox_auto_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BwrapSandbox, "available", lambda self: False)
    monkeypatch.setattr(UnshareSandbox, "available", lambda self: False)
    with pytest.raises(SandboxUnavailable):
        get_sandbox("auto")


def test_get_sandbox_rejects_network_none() -> None:
    with pytest.raises(SandboxUnavailable):
        get_sandbox("unshare", network="none")
