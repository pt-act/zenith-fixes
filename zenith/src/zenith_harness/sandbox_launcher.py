"""Launcher used by the unshare sandbox backend.

Runs inside a fresh user/mount/pid namespace created by util-linux `unshare`.
Builds a minimal root, binds the workspace read-write, pivots, then execs the
target. Kept deliberately small; bwrap is the preferred production backend.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")


def _mount(source: str, target: Path, *, read_only: bool) -> None:
    target.mkdir(parents=True, exist_ok=True)
    _run(["mount", "--bind", source, str(target)])
    if read_only:
        _run(["mount", "-o", "remount,bind,ro", str(target)])


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    root = Path(config["root"])
    workspace = Path(config["workspace"])
    home = Path(config["home"])
    cwd = Path(config["cwd"])
    argv: list[str] = config["argv"]
    env: dict[str, str] = config["env"]

    mount_bin = shutil.which("mount")
    pivot_bin = shutil.which("pivot_root") or next(
        (
            candidate
            for candidate in ("/usr/sbin/pivot_root", "/sbin/pivot_root")
            if Path(candidate).exists()
        ),
        None,
    )
    umount_bin = shutil.which("umount")
    if not mount_bin or not pivot_bin or not umount_bin:
        raise RuntimeError("mount/pivot_root/umount not found on the host")

    # A private mount tree: nothing the child does can propagate to the host.
    _run([mount_bin, "--make-rprivate", "/"])
    _run([mount_bin, "-t", "tmpfs", "tmpfs", str(root)])

    for name in ("usr", "usr/local", "bin", "sbin", "lib", "lib64", "lib32", "etc"):
        source = Path("/" + name)
        if source.exists():
            _mount(str(source), root / name, read_only=True)

    for raw in config.get("extra_ro_paths") or []:
        source = Path(raw)
        if not source.exists():
            continue
        target = root / raw.lstrip("/")
        if source.is_dir():
            _mount(str(source), target, read_only=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch(exist_ok=True)
            _run([mount_bin, "--bind", str(source), str(target)])
            _run([mount_bin, "-o", "remount,bind,ro", str(target)])

    _mount(str(workspace), root / str(workspace).lstrip("/"), read_only=False)
    _mount(str(home), root / str(home).lstrip("/"), read_only=False)

    (root / "proc").mkdir(parents=True, exist_ok=True)
    _run([mount_bin, "-t", "proc", "proc", str(root / "proc")])
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    _run([mount_bin, "-t", "tmpfs", "tmpfs", str(root / "tmp")])

    dev = root / "dev"
    dev.mkdir(parents=True, exist_ok=True)
    for name in ("null", "zero", "urandom", "tty"):
        source = Path("/dev") / name
        if not source.exists():
            continue
        target = dev / name
        target.touch(exist_ok=True)
        try:
            _run([mount_bin, "--bind", str(source), str(target)])
        except RuntimeError:
            pass

    (root / ".old").mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    _run([pivot_bin, ".", ".old"])
    try:
        _run([umount_bin, "-l", "/.old"])
    except RuntimeError:
        pass

    os.chdir(cwd)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"zenith sandbox launcher: {exc}", file=sys.stderr)
        raise SystemExit(127)
