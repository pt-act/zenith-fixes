"""Regression tests for the audit findings not covered by the first pass.

Covers H-9 (shell-string ACP launch), M-1 (attention lifecycle), H-5
(workspace mutation opt-in), H-8 (terminal reader cleanup) and M-9 (single
CLI/MCP controller construction).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

import pytest

from zenith_harness.config import HarnessConfig
from zenith_harness.models import AttentionItemInternal
from zenith_harness.storage import ProjectStore


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    home = tmp_path / "zenith-home"
    home.mkdir()
    monkeypatch.setenv("ZENITH_HOME", str(home))
    monkeypatch.delenv("ZENITH_PROJECT_BUCKET_DIR", raising=False)
    monkeypatch.delenv("ZENITH_WORKSPACE_SHIMS", raising=False)
    monkeypatch.delenv("ZENITH_REPLACE_BOOTSTRAP_SKILLS", raising=False)
    return ProjectStore(HarnessConfig.discover())


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("user repo\n", encoding="utf-8")
    return ws


# ---------------------------------------------------------------------------
# H-9 — the ACP command must reach exec as argv, never a shell
# ---------------------------------------------------------------------------


async def test_acp_spawn_does_not_interpret_shell_metacharacters(
    tmp_path: Path,
) -> None:
    from zenith_harness.acp_runner import _spawn_acp_process

    marker = tmp_path / "pwned"
    command = f"{sys.executable} -c \"print('ok')\" ; touch {marker}"
    process = await _spawn_acp_process(
        command, cwd=str(tmp_path), env=os.environ.copy()
    )
    stdout, _ = await process.communicate()

    assert b"ok" in stdout
    assert not marker.exists(), "shell metacharacters were interpreted"


async def test_acp_spawn_rejects_an_empty_command(tmp_path: Path) -> None:
    from zenith_harness.acp_runner import ACPError, _spawn_acp_process

    with pytest.raises(ACPError):
        await _spawn_acp_process("   ", cwd=str(tmp_path), env=os.environ.copy())


def test_acp_argv_keeps_quoted_arguments_together() -> None:
    from zenith_harness.acp_runner import _acp_argv

    assert _acp_argv('codex-acp -c model="gpt 5"') == [
        "codex-acp",
        "-c",
        "model=gpt 5",
    ]


# ---------------------------------------------------------------------------
# M-1 — the attention queue is bounded, deduped, and consumed explicitly
# ---------------------------------------------------------------------------


def _item(item_id: str, kind: str, node: str = "N1") -> AttentionItemInternal:
    return AttentionItemInternal(
        id=item_id,
        kind=kind,  # type: ignore[arg-type]
        mission_id="mission-001",
        node_id=node,
        report="report",
    )


def test_attention_queue_is_capped_and_keeps_failures() -> None:
    from zenith_harness.attention import (
        MAX_OPEN_ATTENTION_ITEMS,
        cap_attention_items,
    )

    checkpoints = [
        _item(f"att-check-{i}", "gate_checkpoint", node=f"G{i}")
        for i in range(MAX_OPEN_ATTENTION_ITEMS + 10)
    ]
    failure = _item("att-failure", "node_failed", node="W1")

    capped = cap_attention_items([*checkpoints, failure])

    assert len(capped) == MAX_OPEN_ATTENTION_ITEMS
    assert capped[-1].id == "att-failure"
    # Overflow must drop the oldest checkpoints, not the actionable failure.
    assert "att-check-0" not in {it.id for it in capped}
    assert "att-failure" in {it.id for it in capped}


def test_attention_queue_dedupes_by_id() -> None:
    from zenith_harness.attention import cap_attention_items

    same = _item("att-same", "node_failed")
    assert [it.id for it in cap_attention_items([same, same])] == ["att-same"]


def test_consume_attention_removes_only_decided_items(store: ProjectStore, tmp_path: Path) -> None:
    record = store.create_project("consume attention", _workspace(tmp_path))
    store.save_attention(record.id, [_item("att-a", "node_failed"), _item("att-b", "gate_failed")])

    store.consume_attention(record.id, {"att-a"})

    remaining = store.load_attention(record.id)
    assert [it.id for it in remaining] == ["att-b"]


# ---------------------------------------------------------------------------
# H-5 — workspace mutation is opt-in
# ---------------------------------------------------------------------------


def test_workspace_shims_are_opt_in(store: ProjectStore, tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    store.create_project("shim opt-in", workspace)

    assert not (workspace / ".agents").exists()
    assert not (workspace / ".claude").exists()
    assert not (workspace / ".codex").exists()
    assert not (workspace / "AGENTS.md").exists()


def test_workspace_shims_are_relative_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zenith-home"
    home.mkdir()
    monkeypatch.setenv("ZENITH_HOME", str(home))
    monkeypatch.setenv("ZENITH_WORKSPACE_SHIMS", "1")
    store = ProjectStore(HarnessConfig.discover())

    workspace = _workspace(tmp_path)
    store.create_project("shim enabled", workspace)

    links = [p for p in workspace.rglob("*") if p.is_symlink()]
    assert links
    for link in links:
        assert not os.readlink(link).startswith("/"), link


def test_bootstrap_skill_dir_is_not_deleted_without_the_delete_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zenith-home"
    home.mkdir()
    monkeypatch.setenv("ZENITH_HOME", str(home))
    monkeypatch.setenv("ZENITH_WORKSPACE_SHIMS", "1")
    monkeypatch.delenv("ZENITH_REPLACE_BOOTSTRAP_SKILLS", raising=False)
    config = HarnessConfig.discover()
    store = ProjectStore(config)

    workspace = _workspace(tmp_path)
    bundled_skills = config.bundled_dir / "skills"
    bundled_skill = next(p for p in sorted(bundled_skills.iterdir()) if p.is_dir())
    shutil.copytree(bundled_skill, workspace / ".agents" / "skills" / bundled_skill.name)

    store.create_project("bootstrap skills", workspace)

    host_skills = workspace / ".agents" / "skills"
    assert host_skills.is_dir()
    assert not host_skills.is_symlink(), "user workspace directory was replaced"


# ---------------------------------------------------------------------------
# H-8 — cleanup must cancel and await terminal reader tasks
# ---------------------------------------------------------------------------


async def test_cleanup_awaits_and_clears_terminal_reader_tasks(tmp_path: Path) -> None:
    from zenith_harness.acp_runner import ACPClient

    class _FakeProcess:
        stdin = None
        stdout = None

    client = ACPClient(
        _FakeProcess(),  # type: ignore[arg-type]
        str(tmp_path),
        allow_terminal=True,
    )
    await client._handle_terminal_create(
        {
            "command": sys.executable,
            "args": ["-c", "import time; time.sleep(30)"],
            "cwd": str(tmp_path),
        }
    )
    readers = set(client._terminal_tasks)
    assert readers

    await asyncio.wait_for(client.cleanup(close_main_process=False), timeout=10)

    assert client._terminal_tasks == set()
    assert client._terminals == {}
    assert all(task.done() for task in readers)


# ---------------------------------------------------------------------------
# M-9 — one controller construction for the CLI and the MCP server
# ---------------------------------------------------------------------------


def test_cli_abort_project_marks_the_project_aborted(
    store: ProjectStore, tmp_path: Path
) -> None:
    from click.testing import CliRunner

    from zenith_harness.cli import cli

    record = store.create_project("abort me", _workspace(tmp_path))

    result = CliRunner().invoke(
        cli, ["abort-project", record.id, "--reason", "test", "--yes"]
    )

    assert result.exit_code == 0, result.output
    assert store.load_state(record.id).state == "aborted"


def test_build_controller_is_shared_by_cli_and_server() -> None:
    from zenith_harness.controller import ProjectController, build_controller

    controller = build_controller(HarnessConfig.discover())
    assert isinstance(controller, ProjectController)
