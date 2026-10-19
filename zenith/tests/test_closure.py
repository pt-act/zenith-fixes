"""Fail-closed closure policy tests (reviewer is a veto, not an authority)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from zenith_harness.closure import (
    AcceptanceResult,
    ClosureCommand,
    ClosureSpec,
    evaluate_closure,
    load_closure_spec,
    run_acceptance,
)
from zenith_harness.config import HarnessConfig
from zenith_harness.controller import ProjectController
from zenith_harness.dispatcher import MockDispatcher, MockTerminalReviewer
from zenith_harness.models import (
    ContractStateEntry,
    ContractStateFile,
    MissionRunning,
    Task,
    TaskList,
    TaskStateFile,
    TerminalReviewHandoff,
    WorkHandoff,
)
from zenith_harness.storage import ProjectStore


@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    home = tmp_path / "zenith-home"
    home.mkdir()
    monkeypatch.setenv("ZENITH_HOME", str(home))
    monkeypatch.delenv("ZENITH_PROJECT_BUCKET_DIR", raising=False)
    return ProjectStore(HarnessConfig.discover())


def _seed(
    store: ProjectStore,
    tmp_path: Path,
    *,
    names: str = "closure",
    task_status: str = "cleared",
    gate_status: str = "cleared",
    assertion: str = "passed",
) -> tuple[str, str, Path]:
    ws = tmp_path / names
    ws.mkdir()
    (ws / "README.md").write_text("product\n", encoding="utf-8")
    record = store.create_project(f"closure {names}", ws)
    mid = "mission-001"
    record.current_mission_id = mid
    store.save_project(record)
    store.save_state(record.id, MissionRunning(mission_id=mid))
    store.save_task_list(
        record.id,
        mid,
        TaskList(
            tasks=[
                Task(id="W1", type="work", body="b", targets=["A1"], skill="s"),
                Task(
                    id="G1",
                    type="gate",
                    body="",
                    targets=["A1"],
                    depends_on=["W1"],
                ),
            ]
        ),
    )
    ts = TaskStateFile()
    ts.set_status("W1", task_status)  # type: ignore[arg-type]
    ts.set_status("G1", gate_status)  # type: ignore[arg-type]
    store.save_task_state(record.id, mid, ts)
    store.save_contract_state(
        record.id,
        mid,
        ContractStateFile(items={"A1": ContractStateEntry(status=assertion)}),  # type: ignore[arg-type]
    )
    return record.id, mid, ws


def _evaluate(
    store: ProjectStore,
    pid: str,
    mid: str,
    ws: Path,
    review: TerminalReviewHandoff,
    *,
    read_paths: set[str] | None = None,
    acceptance: AcceptanceResult | None = None,
):
    return evaluate_closure(
        state=store.load_state(pid),
        task_list=store.load_task_list(pid, mid),
        task_state=store.load_task_state(pid, mid),
        contract_state=store.load_contract_state(pid, mid),
        review=review,
        workspace=ws,
        read_paths=read_paths,
        acceptance=acceptance,
        open_attention=0,
    )


def test_clean_mission_with_read_evidence_is_allowed(store: ProjectStore, tmp_path: Path) -> None:
    pid, mid, ws = _seed(store, tmp_path)
    review = TerminalReviewHandoff(done=True, report="clean", evidence=["README.md"])
    decision = _evaluate(
        store, pid, mid, ws, review, read_paths={str((ws / "README.md").resolve())}
    )
    assert decision.allowed, decision.blockers


def test_pending_assertion_blocks_even_when_reviewer_says_done(
    store: ProjectStore, tmp_path: Path
) -> None:
    pid, mid, ws = _seed(store, tmp_path, assertion="pending")
    review = TerminalReviewHandoff(done=True, report="clean", evidence=["README.md"])
    decision = _evaluate(
        store, pid, mid, ws, review, read_paths={str((ws / "README.md").resolve())}
    )
    assert not decision.allowed
    assert any("assertion A1 is pending" in b for b in decision.blockers)


def test_evidence_not_read_through_acp_blocks(store: ProjectStore, tmp_path: Path) -> None:
    pid, mid, ws = _seed(store, tmp_path)
    review = TerminalReviewHandoff(done=True, report="clean", evidence=["README.md"])
    decision = _evaluate(store, pid, mid, ws, review, read_paths=set())
    assert not decision.allowed
    assert any("did not read it through ACP" in b for b in decision.blockers)


def test_acceptance_failure_blocks(store: ProjectStore, tmp_path: Path) -> None:
    pid, mid, ws = _seed(store, tmp_path)
    review = TerminalReviewHandoff(done=True, report="clean", evidence=["README.md"])
    acceptance = AcceptanceResult(passed=False, reports=["FAIL pytest: exit 1"])
    decision = _evaluate(
        store,
        pid,
        mid,
        ws,
        review,
        read_paths={str((ws / "README.md").resolve())},
        acceptance=acceptance,
    )
    assert not decision.allowed
    assert any("acceptance" in b for b in decision.blockers)


def test_run_acceptance_pass_and_fail(tmp_path: Path) -> None:
    ok = run_acceptance(
        ClosureSpec(commands=[ClosureCommand(argv=[sys.executable, "-c", "pass"])]),
        tmp_path,
    )
    assert ok.passed
    bad = run_acceptance(
        ClosureSpec(
            commands=[ClosureCommand(argv=[sys.executable, "-c", "raise SystemExit(3)"])]
        ),
        tmp_path,
    )
    assert not bad.passed
    assert "exit 3" in bad.summary()


def test_load_closure_spec_rejects_invalid(store: ProjectStore, tmp_path: Path) -> None:
    pid, mid, _ = _seed(store, tmp_path)
    path = store.mission_dir(pid, mid) / "closure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(Exception):
        load_closure_spec(store, pid, mid)


def _controller(store: ProjectStore, reviewer) -> ProjectController:  # noqa: ANN001
    return ProjectController(
        store.config,
        MockDispatcher(lambda r: WorkHandoff(node_id=r.task.id, done=True, report="")),
        reviewer,
        store=store,
    )


def _empty_mission(store: ProjectStore, tmp_path: Path) -> tuple[str, str, Path]:
    ws = tmp_path / "empty"
    ws.mkdir()
    (ws / "README.md").write_text("product\n", encoding="utf-8")
    record = store.create_project("empty closure", ws)
    mid = "mission-001"
    record.current_mission_id = mid
    store.save_project(record)
    store.save_state(record.id, MissionRunning(mission_id=mid))
    store.save_task_list(record.id, mid, TaskList(tasks=[]))
    store.save_task_state(record.id, mid, TaskStateFile())
    store.save_contract_state(record.id, mid, ContractStateFile())
    return record.id, mid, ws


def test_coordinator_acceptance_failure_blocks_done(store: ProjectStore, tmp_path: Path) -> None:
    pid, mid, ws = _empty_mission(store, tmp_path)
    closure = store.mission_dir(pid, mid) / "closure.json"
    closure.parent.mkdir(parents=True, exist_ok=True)
    closure.write_text(
        json.dumps(
            {"commands": [{"argv": [sys.executable, "-c", "raise SystemExit(1)"]}]}
        ),
        encoding="utf-8",
    )
    reviewer = MockTerminalReviewer(
        TerminalReviewHandoff(done=True, report="clean", evidence=["README.md"])
    )
    env = _controller(store, reviewer).end_mission(pid)
    assert env.state.state == "attention_needed"
    assert store.load_attention(pid)[0].kind == "terminal_review"
    assert "acceptance command failed" in store.load_attention(pid)[0].report


def test_coordinator_acceptance_pass_seals_done(store: ProjectStore, tmp_path: Path) -> None:
    pid, mid, _ = _empty_mission(store, tmp_path)
    closure = store.mission_dir(pid, mid) / "closure.json"
    closure.parent.mkdir(parents=True, exist_ok=True)
    closure.write_text(
        json.dumps(
            {"commands": [{"argv": [sys.executable, "-c", "pass"]}]}
        ),
        encoding="utf-8",
    )
    reviewer = MockTerminalReviewer(
        TerminalReviewHandoff(done=True, report="clean", evidence=["README.md"])
    )
    env = _controller(store, reviewer).end_mission(pid)
    assert env.state.state == "done"


def test_acp_read_log_records_served_paths(store: ProjectStore, tmp_path: Path) -> None:
    from zenith_harness.acp_runner import ACPClient

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "README.md").write_text("product\n", encoding="utf-8")

    class FakeProcess:
        stdin = None
        stdout = None

    client = ACPClient(FakeProcess(), str(ws), allow_terminal=False)  # type: ignore[arg-type]
    result = client._handle_fs_read({"path": "README.md"})
    assert result["content"] == "product\n"
    assert str((ws / "README.md").resolve()) in client.read_paths


def test_read_paths_are_not_serialized_or_forgeable() -> None:
    review = TerminalReviewHandoff(done=True, evidence=["README.md"])
    review.set_read_paths({"/tmp/a", "/tmp/b"})
    assert review.read_paths == ["/tmp/a", "/tmp/b"]
    assert "read_paths" not in review.model_dump(mode="json")
    with pytest.raises(Exception):
        TerminalReviewHandoff.model_validate(
            {"done": True, "read_paths": ["/forged"]}
        )
