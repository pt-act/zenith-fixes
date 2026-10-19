"""Regression tests for the second audit's remediations (2026-10-19).

Each test names the finding it pins (C-01, H-01..H-04, M-01..M-11, L-01...).
The initial audit's probes are the acceptance tests; these keep the fixes from
regressing without requiring the probe harness.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from zenith_harness.attention import cap_attention_items
from zenith_harness.config import HarnessConfig
from zenith_harness.controller import ProjectController, ToolError
from zenith_harness.dispatcher import MockDispatcher, MockTerminalReviewer
from zenith_harness.models import (
    AttentionItem,
    AttentionItemInternal,
    AttentionNeeded,
    ContractStateEntry,
    ContractStateFile,
    Decision,
    Draft,
    MissionRunning,
    Task,
    TaskList,
    TaskStateFile,
    TerminalReviewHandoff,
    WorkHandoff,
)
from zenith_harness.providers import PROVIDERS
from zenith_harness.storage import ProjectBusy, ProjectStore


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    home = tmp_path / "zenith-home"
    home.mkdir()
    monkeypatch.setenv("ZENITH_HOME", str(home))
    monkeypatch.delenv("ZENITH_PROJECT_BUCKET_DIR", raising=False)
    return ProjectStore(HarnessConfig.discover())


def _controller(
    store: ProjectStore,
    *,
    reviewer=None,
    dispatcher=None,
) -> ProjectController:
    return ProjectController(
        store.config,
        dispatcher or MockDispatcher(lambda r: WorkHandoff(node_id=r.task.id, done=True, report="")),
        reviewer or MockTerminalReviewer(TerminalReviewHandoff(done=True, report="")),
        store=store,
    )


def _seed_attention(
    store: ProjectStore,
    tmp_path: Path,
    *,
    status: str = "failed",
    kind: str = "node_failed",
) -> tuple[str, str, str]:
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    record = store.create_project("attention", workspace)
    mid = "mission-001"
    record.current_mission_id = mid
    store.save_project(record)
    store.save_state(record.id, MissionRunning(mission_id=mid))
    store.save_task_list(
        record.id,
        mid,
        TaskList(tasks=[Task(id="W1", type="work", body="b", targets=["A1"], skill="s")]),
    )
    ts = TaskStateFile()
    ts.set_status("W1", status)  # type: ignore[arg-type]
    store.save_task_state(record.id, mid, ts)
    store.save_contract_state(
        record.id, mid, ContractStateFile(items={"A1": ContractStateEntry()})
    )
    item = AttentionItemInternal(
        id="att-1",
        kind=kind,  # type: ignore[arg-type]
        mission_id=mid,
        node_id="W1",
        report="failed",
    )
    store.save_attention(record.id, [item])
    store.save_state(
        record.id,
        AttentionNeeded(items=[AttentionItem(id=item.id, report=item.report, kind=item.kind, mission_id=mid)]),
    )
    return record.id, mid, item.id


# ---------------------------------------------------------------------------
# C-01 — ACP agent environment is allowlisted
# ---------------------------------------------------------------------------


def test_acp_env_does_not_inherit_unrelated_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    from zenith_harness.acp_runner import _acp_subprocess_env

    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-sentinel")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-sentinel")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    env = _acp_subprocess_env(PROVIDERS["codex"])
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert env["OPENAI_API_KEY"] == "provider-key"


def test_acp_env_opt_in_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    from zenith_harness.acp_runner import _acp_subprocess_env

    monkeypatch.setenv("PRIVATE_PROXY_TOKEN", "proxy-token")
    monkeypatch.setenv("ZENITH_ACP_ENV_ALLOW", "PRIVATE_PROXY_TOKEN")
    env = _acp_subprocess_env(PROVIDERS["claude"])
    assert env["PRIVATE_PROXY_TOKEN"] == "proxy-token"


# ---------------------------------------------------------------------------
# H-01 — the reviewer brief is inside the reviewer's workspace
# ---------------------------------------------------------------------------


def test_reviewer_brief_is_inside_workspace(store: ProjectStore, tmp_path: Path) -> None:
    from zenith_harness.acp_runner import ACPNodeRunner
    from zenith_harness.assets import AssetLoader

    ws = tmp_path / "ws"
    ws.mkdir()
    record = store.create_project("Build a page.", ws)
    runner = ACPNodeRunner(config=store.config, loader=AssetLoader(store.config))
    prompt, review_dir = runner._render_terminal_reviewer_prompts(
        project_bucket=str(store.zenith_dir(record.id)),
        workspace_dir=str(ws),
    )
    brief = review_dir / "brief.txt"
    assert brief.is_relative_to(ws.resolve())
    assert str(brief) in prompt
    assert "UNTRUSTED USER REQUEST" in brief.read_text()
    import shutil

    shutil.rmtree(review_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# H-02 — a failed decision commit rolls back
# ---------------------------------------------------------------------------


def test_decide_attention_rolls_back_on_io_failure(store: ProjectStore, tmp_path: Path) -> None:
    pid, mid, item_id = _seed_attention(store, tmp_path)
    controller = _controller(store)
    original = store.save_task_state
    failed = {"once": False}

    def flaky(project_id, mission_id, task_state):
        if not failed["once"]:
            failed["once"] = True
            raise OSError("disk full")
        return original(project_id, mission_id, task_state)

    store.save_task_state = flaky  # type: ignore[method-assign]
    with pytest.raises(OSError):
        controller.decide_attention(pid, [Decision(item_id=item_id, action="continue")])

    # The pre-image was restored: the task is still failed and the item is open.
    assert store.load_task_state(pid, mid).status_of("W1") == "failed"
    assert len(store.load_attention(pid)) == 1
    assert not store.decision_journal_path(pid).exists()

    # Restore the writer and retry the same decision.
    store.save_task_state = original  # type: ignore[method-assign]
    env = controller.decide_attention(pid, [Decision(item_id=item_id, action="continue")])
    assert env.state.state == "mission_running"
    assert store.load_task_state(pid, mid).status_of("W1") == "cleared"


# ---------------------------------------------------------------------------
# H-03 — a reviewer crash becomes retryable attention, not Failed
# ---------------------------------------------------------------------------


class _CrashingReviewer:
    def review(self, project_id: str, mission_id: str, spawn_ts: str):
        raise RuntimeError("adapter died")


def test_reviewer_crash_raises_attention(store: ProjectStore, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    record = store.create_project("review crash", ws)
    mid = "mission-001"
    record.current_mission_id = mid
    store.save_project(record)
    store.save_state(record.id, MissionRunning(mission_id=mid))
    store.save_task_list(record.id, mid, TaskList(tasks=[]))
    store.save_task_state(record.id, mid, TaskStateFile())
    store.save_contract_state(record.id, mid, ContractStateFile())

    controller = _controller(store, reviewer=_CrashingReviewer())  # type: ignore[arg-type]
    env = controller.end_mission(record.id)
    assert env.state.state == "attention_needed"
    items = store.load_attention(record.id)
    assert len(items) == 1
    assert items[0].kind == "terminal_review"
    assert "crashed" in items[0].report


# ---------------------------------------------------------------------------
# H-04 — cross-process lock
# ---------------------------------------------------------------------------


def test_project_lock_excludes_a_second_holder(store: ProjectStore, tmp_path: Path) -> None:
    pid = store.create_project("lock", tmp_path).id
    with store.project_lock(pid):
        with pytest.raises(ProjectBusy):
            with store.project_lock(pid, timeout=0.2):
                pass


# ---------------------------------------------------------------------------
# M-01 / M-02 — corrupt artifacts are ToolErrors and abort still works
# ---------------------------------------------------------------------------


def test_corrupt_state_is_tool_error_and_abort_works(store: ProjectStore, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    record = store.create_project("corrupt", ws)
    store.save_state(record.id, Draft())
    (store.zenith_runtime_dir(record.id) / "state.json").write_text("{not json")

    controller = _controller(store)
    with pytest.raises(ToolError) as exc:
        controller.inspect_project(record.id)
    assert exc.value.code == "corrupt_state"

    controller.abort_project(record.id, "recover")
    state = store.load_state(record.id)
    assert state is not None and state.state == "aborted"


def test_unreadable_attempt_is_quarantined(store: ProjectStore, tmp_path: Path) -> None:
    pid, mid, _ = _seed_attention(store, tmp_path, status="running")
    path = store.attempt_path(pid, mid, "2026-01-01T00-00-00Z", "W1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe\x00broken")
    handoff = store.read_attempt(pid, mid, "2026-01-01T00-00-00Z", "W1")
    assert handoff is not None and handoff.done is False
    assert not path.exists() or path.with_suffix(".json.corrupt").exists()


# ---------------------------------------------------------------------------
# M-06 — start_project is idempotent
# ---------------------------------------------------------------------------


def test_start_project_is_idempotent(store: ProjectStore, tmp_path: Path) -> None:
    controller = _controller(store)
    ws = tmp_path / "ws"
    ws.mkdir()
    first = controller.start_project("same brief", str(ws))
    second = controller.start_project("same brief", str(ws))
    assert first.projectId == second.projectId
    third = controller.start_project("same brief", str(ws), new_project=True)
    assert third.projectId != first.projectId


# ---------------------------------------------------------------------------
# M-10 — the attention cap never drops actionable failures
# ---------------------------------------------------------------------------


def test_acp_client_records_read_tool_calls_as_evidence(
    tmp_path: Path,
) -> None:
    from zenith_harness.acp_runner import ACPClient

    client = object.__new__(ACPClient)
    client._workspace_root = tmp_path.resolve()
    client.read_paths = set()
    client._record_read_tool_calls(
        {
            "update": {
                "sessionUpdate": "tool_call",
                "title": "Read File",
                "kind": "read",
                "rawInput": {"path": "README.md"},
            }
        }
    )
    client._record_read_tool_calls(
        {
            "update": {
                "sessionUpdate": "tool_call",
                "title": "Read File",
                "kind": "read",
                "rawInput": {"path": "/etc/passwd"},
            }
        }
    )
    assert str((tmp_path / "README.md").resolve()) in client.read_paths
    assert "/etc/passwd" not in client.read_paths


def test_attention_cap_keeps_actionable_items() -> None:
    failures = [
        AttentionItemInternal(
            id=f"att-f{i}",
            kind="node_failed",
            mission_id="m",
            node_id=f"W{i}",
            report="x",
        )
        for i in range(70)
    ]
    assert len(cap_attention_items(failures)) == 70
    checkpoints = [
        AttentionItemInternal(
            id=f"att-c{i}",
            kind="gate_checkpoint",
            mission_id="m",
            node_id=f"G{i}",
            report="x",
        )
        for i in range(70)
    ]
    assert len(cap_attention_items(checkpoints)) == 64


# ---------------------------------------------------------------------------
# M-09 — agent text is capped
# ---------------------------------------------------------------------------


def test_agent_report_is_truncated() -> None:
    handoff = WorkHandoff(node_id="W1", done=True, report="x" * 400_000)
    assert len(handoff.report) <= 256 * 1024
    assert "truncated by Zenith" in handoff.report


# ---------------------------------------------------------------------------
# M-11 — reviewer done=true requires evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewer_done_requires_evidence(tmp_path: Path, monkeypatch) -> None:
    from fastmcp.exceptions import ToolError as McpToolError

    from zenith_harness.server import create_terminal_reviewer_server

    monkeypatch.setenv("ZENITH_TERMINAL_REVIEW_PATH", str(tmp_path / "review.json"))
    server = create_terminal_reviewer_server()
    with pytest.raises(McpToolError):
        await server.call_tool("submit_terminal_review", {"done": True, "report": "clean"})


# ---------------------------------------------------------------------------
# L-01 — Windows command splitting preserves backslashes
# ---------------------------------------------------------------------------


def test_windows_command_split_preserves_backslashes() -> None:
    from zenith_harness.acp_runner import _split_command

    argv = _split_command(r'C:\tools\codex-acp.cmd -c model=x', posix=False)
    assert argv == [r"C:\tools\codex-acp.cmd", "-c", "model=x"]
    quoted = _split_command(
        r'"C:\Program Files\codex-acp.cmd" -c model="gpt 5"', posix=False
    )
    assert quoted == [
        r"C:\Program Files\codex-acp.cmd",
        "-c",
        "model=gpt 5",
    ]


# ---------------------------------------------------------------------------
# L-06 — the CLI does not duplicate the aborted label
# ---------------------------------------------------------------------------


def test_aborted_state_label_is_not_duplicated() -> None:
    from zenith_harness.cli import _state_label
    from zenith_harness.models import Aborted

    label = _state_label(Aborted(reason="x"))
    assert label == "aborted — mission stopped"


# ---------------------------------------------------------------------------
# H-02 — an interrupted journal is recovered on the next tool call
# ---------------------------------------------------------------------------


def test_interrupted_journal_is_recovered(store: ProjectStore, tmp_path: Path) -> None:
    pid, mid, item_id = _seed_attention(store, tmp_path)
    state_path = store.zenith_runtime_dir(pid) / "state.json"
    store.begin_decision_journal(pid, [state_path])
    # Simulate a crash mid-commit: mutate state after the snapshot.
    store.save_state(pid, Draft())
    _controller(store).inspect_project(pid)
    recovered = store.load_state(pid)
    assert recovered is not None and recovered.state == "attention_needed"
    assert not store.decision_journal_path(pid).exists()


# ---------------------------------------------------------------------------
# M-11 — the coordinator rejects a done verdict with a bogus evidence path
# ---------------------------------------------------------------------------


def test_done_with_bogus_evidence_is_rejected(store: ProjectStore, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    record = store.create_project("evidence", ws)
    mid = "mission-001"
    record.current_mission_id = mid
    store.save_project(record)
    store.save_state(record.id, MissionRunning(mission_id=mid))
    store.save_task_list(record.id, mid, TaskList(tasks=[]))
    store.save_task_state(record.id, mid, TaskStateFile())
    store.save_contract_state(record.id, mid, ContractStateFile())

    reviewer = MockTerminalReviewer(
        TerminalReviewHandoff(done=True, report="all clear", evidence=["does-not-exist.txt"])
    )
    env = _controller(store, reviewer=reviewer).end_mission(record.id)
    assert env.state.state == "attention_needed"
    items = store.load_attention(record.id)
    assert "evidence" in items[0].report.lower()


# ---------------------------------------------------------------------------
# L-05 — the CLI abort prompt is a real confirmation
# ---------------------------------------------------------------------------


def test_abort_requires_confirmation(store: ProjectStore, tmp_path: Path) -> None:
    from click.testing import CliRunner

    from zenith_harness.cli import cli

    ws = tmp_path / "ws"
    ws.mkdir()
    record = store.create_project("abort prompt", ws)
    store.save_state(record.id, Draft())
    runner = CliRunner()
    declined = runner.invoke(
        cli,
        ["abort-project", record.id, "--reason", "test"],
        input="n\n",
    )
    assert declined.exit_code != 0
    state = store.load_state(record.id)
    assert state is not None and state.state == "draft"


# ---------------------------------------------------------------------------
# M-03 — the worker server binds and reports its own port
# ---------------------------------------------------------------------------


def test_worker_server_reports_bound_port(tmp_path: Path) -> None:
    port_file = tmp_path / "port"
    env = os.environ.copy()
    env["ZENITH_MCP_AUTH_TOKEN"] = "test-token"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "zenith_harness",
            "--mode",
            "worker",
            "--transport",
            "streamable-http",
            "--port",
            "0",
            "--port-file",
            str(port_file),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not port_file.exists():
            time.sleep(0.05)
        assert port_file.exists(), proc.stderr.read().decode() if proc.stderr else ""
        port = int(port_file.read_text().strip())
        assert 0 < port < 65536
    finally:
        proc.terminate()
        proc.wait(timeout=5)
