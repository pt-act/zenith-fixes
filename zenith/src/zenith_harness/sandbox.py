"""Pluggable OS-level sandbox for ACP agents and their terminal commands.

Phase 1 backends:

- ``bwrap``: bubblewrap, when installed. Preferred: it handles user/mount/pid
  namespaces, read-only binds, tmpfs, `/proc`, `/dev`, and `--die-with-parent`.
- ``unshare``: Linux fallback built from util-linux ``unshare`` +
  ``pivot_root``. Proven on hosts with unprivileged user namespaces. No
  seccomp and no network namespace by itself.
- ``none``: explicit opt-out. Never selected by ``auto``.

The mission workspace is bound read-write at its existing absolute path. Host
system trees are read-only. The host home and ``$ZENITH_HOME`` are not bound;
the agent gets a per-node home containing only its provider's config.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

SANDBOX_BACKENDS = ("auto", "bwrap", "unshare", "none")


class SandboxUnavailable(RuntimeError):
    """A requested/auto sandbox cannot be established on this host."""


class Sandbox(Protocol):
    name: str

    def wrap(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        workspace: Path,
        home: Path,
        extra_ro_paths: tuple[str, ...] = (),
    ) -> list[str]: ...


class NoSandbox:
    name = "none"

    def wrap(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        workspace: Path,
        home: Path,
        extra_ro_paths: tuple[str, ...] = (),
    ) -> list[str]:
        return list(argv)


def _existing(path: str) -> bool:
    return Path(path).exists()


def _ro_bind_paths(extra_ro_paths: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for candidate in (
        "/usr",
        "/usr/local",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/lib32",
        "/etc",
        *extra_ro_paths,
    ):
        if candidate and _existing(candidate) and candidate not in paths:
            paths.append(candidate)
    return paths


class BwrapSandbox:
    name = "bwrap"

    def __init__(self, binary: str | None = None) -> None:
        self.binary = binary or shutil.which("bwrap") or ""

    def available(self) -> bool:
        return bool(self.binary) and Path(self.binary).exists()

    def wrap(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        workspace: Path,
        home: Path,
        extra_ro_paths: tuple[str, ...] = (),
    ) -> list[str]:
        if not self.available():
            raise SandboxUnavailable("bwrap is not installed")
        workspace = workspace.expanduser().resolve()
        home = home.expanduser().resolve()
        cwd = cwd.expanduser().resolve()
        if not cwd.is_relative_to(workspace):
            raise SandboxUnavailable(f"cwd {cwd} is outside the workspace")
        cmd = [
            self.binary,
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-user",
            "--uid",
            "0",
            "--gid",
            "0",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        for path in _ro_bind_paths(extra_ro_paths):
            cmd += ["--ro-bind-try", path, path]
        cmd += ["--bind", str(workspace), str(workspace)]
        cmd += ["--bind", str(home), str(home)]
        cmd += ["--chdir", str(cwd), "--clearenv"]
        for key, value in env.items():
            cmd += ["--setenv", key, value]
        cmd += ["--", *argv]
        return cmd


class UnshareSandbox:
    name = "unshare"

    def __init__(self, unshare: str | None = None) -> None:
        self.unshare = unshare or shutil.which("unshare") or ""
        self.pivot_root = (
            shutil.which("pivot_root")
            or next(
                (
                    candidate
                    for candidate in (
                        "/usr/sbin/pivot_root",
                        "/sbin/pivot_root",
                    )
                    if Path(candidate).exists()
                ),
                "",
            )
        )

    def available(self) -> bool:
        if not (sys.platform.startswith("linux") and self.unshare and self.pivot_root):
            return False
        try:
            probe = subprocess.run(
                [self.unshare, "--user", "--map-root-user", "true"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return probe.returncode == 0

    def wrap(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        workspace: Path,
        home: Path,
        extra_ro_paths: tuple[str, ...] = (),
    ) -> list[str]:
        if not self.available():
            raise SandboxUnavailable(
                "unshare sandbox unavailable (need Linux user namespaces, "
                "unshare, and pivot_root)"
            )
        workspace = workspace.expanduser().resolve()
        home = home.expanduser().resolve()
        cwd = cwd.expanduser().resolve()
        if not cwd.is_relative_to(workspace):
            raise SandboxUnavailable(f"cwd {cwd} is outside the workspace")
        launch_dir = home.parent
        launch_dir.mkdir(parents=True, exist_ok=True)
        root = launch_dir / "root"
        root.mkdir(parents=True, exist_ok=True)
        config = {
            "argv": argv,
            "cwd": str(cwd),
            "env": env,
            "workspace": str(workspace),
            "home": str(home),
            "root": str(root),
            "extra_ro_paths": list(extra_ro_paths),
        }
        config_path = launch_dir / "launch.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return [
            self.unshare,
            "--user",
            "--map-root-user",
            "--mount",
            "--pid",
            "--fork",
            "--kill-child",
            sys.executable,
            "-m",
            "zenith_harness.sandbox_launcher",
            str(config_path),
        ]


def _copy_tree_or_file(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    try:
        if source.is_dir():
            shutil.copytree(
                source,
                destination,
                dirs_exist_ok=True,
                symlinks=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "node_modules"),
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    except OSError as exc:
        logger.debug("sandbox home copy %s -> %s failed: %s", source, destination, exc)


def prepare_node_home(provider_name: str | None, destination: Path) -> Path:
    """Create a 0700 home with only the selected provider's config.

    The agent still receives its provider key through the allowlisted env; this
    keeps unrelated files (~/.ssh, ~/.aws, other profiles) out of the sandbox.
    """
    destination.mkdir(parents=True, exist_ok=True)
    try:
        destination.chmod(0o700)
    except OSError:
        pass
    source_home = Path.home()
    if provider_name == "codex":
        _copy_tree_or_file(source_home / ".codex", destination / ".codex")
    elif provider_name == "claude":
        _copy_tree_or_file(source_home / ".claude", destination / ".claude")
        _copy_tree_or_file(source_home / ".claude.json", destination / ".claude.json")
    elif provider_name == "hermes":
        hermes = destination / ".hermes"
        hermes.mkdir(parents=True, exist_ok=True)
        for name in ("config.yaml", ".env"):
            _copy_tree_or_file(source_home / ".hermes" / name, hermes / name)
        for name in ("skills", "plugins"):
            _copy_tree_or_file(source_home / ".hermes" / name, hermes / name)
    return destination


def sandbox_env_for_provider(
    provider_name: str | None, home: Path, env: dict[str, str]
) -> dict[str, str]:
    """Point the provider at the per-node home, not the real one."""
    result = dict(env)
    result["HOME"] = str(home)
    if provider_name == "codex":
        result["CODEX_HOME"] = str(home / ".codex")
    elif provider_name == "claude":
        result["CLAUDE_CONFIG_DIR"] = str(home / ".claude")
    elif provider_name == "hermes":
        result["HERMES_HOME"] = str(home / ".hermes")
    return result


def get_sandbox(backend: str | None = None, *, network: str | None = None) -> Sandbox:
    """Resolve ZENITH_SANDBOX. ``auto`` fails closed when nothing is usable."""
    choice = (backend or os.environ.get("ZENITH_SANDBOX", "none")).strip().lower()
    if choice not in SANDBOX_BACKENDS:
        raise SandboxUnavailable(
            f"unknown ZENITH_SANDBOX={choice!r}; expected one of "
            f"{', '.join(SANDBOX_BACKENDS)}"
        )
    if choice == "none":
        return NoSandbox()
    if network not in (None, "host"):
        raise SandboxUnavailable(
            "ZENITH_SANDBOX_NETWORK=none is not supported by the phase-1 "
            "backends; the worker MCP server is on host loopback. Use a "
            "container backend for network isolation."
        )
    if choice == "bwrap":
        bwrap_sandbox = BwrapSandbox()
        if not bwrap_sandbox.available():
            raise SandboxUnavailable("ZENITH_SANDBOX=bwrap but bwrap is not installed")
        return bwrap_sandbox
    if choice == "unshare":
        unshare_sandbox = UnshareSandbox()
        if not unshare_sandbox.available():
            raise SandboxUnavailable("ZENITH_SANDBOX=unshare but it is not usable here")
        return unshare_sandbox
    # auto: prefer bwrap, then unshare; never silently run unsandboxed.
    bwrap = BwrapSandbox()
    if bwrap.available():
        return bwrap
    unshare = UnshareSandbox()
    if unshare.available():
        return unshare
    raise SandboxUnavailable(
        "ZENITH_SANDBOX=auto but neither bwrap nor unshare is usable; "
        "set ZENITH_SANDBOX=none only if you accept unsandboxed execution"
    )


__all__ = [
    "BwrapSandbox",
    "NoSandbox",
    "Sandbox",
    "SandboxUnavailable",
    "UnshareSandbox",
    "get_sandbox",
    "prepare_node_home",
    "sandbox_env_for_provider",
]
