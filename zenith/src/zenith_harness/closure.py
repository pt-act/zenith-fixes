"""Fail-closed mission closure.

The terminal reviewer is untrusted text in an instruction position. It may
veto a close, but it must not be the sole authority that grants one. This
module evaluates machine-checkable closure criteria plus the reviewer's
verdict; any blocker keeps the mission open.

A `closure.json` in the mission directory may declare acceptance commands:

    {"commands": [{"argv": ["uv", "run", "--frozen", "pytest", "-q"],
                   "timeout": 600}],
     "require_clean_git": true}

Commands are authored by the trusted orchestrator, run without a shell, with a
credential-free environment and a timeout.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    ContractStateFile,
    MissionRunning,
    ProjectState,
    TaskList,
    TaskStateFile,
    TerminalReviewHandoff,
)
from .storage import CorruptArtifact, ProjectStore

MAX_ACCEPTANCE_OUTPUT = 4000


class ClosureCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    argv: list[str] = Field(min_length=1)
    timeout: float = Field(default=600, gt=0, le=3600)


class ClosureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commands: list[ClosureCommand] = Field(default_factory=list)
    require_clean_git: bool = False


@dataclass
class AcceptanceResult:
    passed: bool
    reports: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return "\n".join(self.reports) if self.reports else "(no acceptance checks)"


@dataclass
class ClosureDecision:
    allowed: bool
    blockers: list[str] = field(default_factory=list)

    def report(self) -> str:
        if not self.blockers:
            return "Closure checks passed."
        return "Closure blocked by machine checks:\n" + "\n".join(
            f"- {blocker}" for blocker in self.blockers
        )


def closure_spec_path(store: ProjectStore, project_id: str, mission_id: str) -> Path:
    return store.mission_dir(project_id, mission_id) / "closure.json"


def load_closure_spec(
    store: ProjectStore, project_id: str, mission_id: str
) -> ClosureSpec | None:
    path = closure_spec_path(store, project_id, mission_id)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        return ClosureSpec.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001 — pydantic/JSON/IO
        raise CorruptArtifact(path, f"invalid closure.json: {exc}") from exc


def _tail(text: str, limit: int = MAX_ACCEPTANCE_OUTPUT) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "...[truncated]...\n" + text[-limit:]


def _acceptance_env() -> dict[str, str]:
    # Reuse the same credential-free allowlist as agent terminal commands.
    from .acp_runner import _terminal_env

    return _terminal_env()


def run_acceptance(spec: ClosureSpec, workspace: Path) -> AcceptanceResult:
    """Run declared closure commands; any failure blocks the mission."""
    reports: list[str] = []
    passed = True
    env = _acceptance_env()
    for command in spec.commands:
        label = " ".join(command.argv)
        try:
            result = subprocess.run(
                command.argv,
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=command.timeout,
                check=False,
            )
        except FileNotFoundError:
            passed = False
            reports.append(f"FAIL {label}: executable not found")
            continue
        except subprocess.TimeoutExpired:
            passed = False
            reports.append(f"FAIL {label}: timed out after {command.timeout:g}s")
            continue
        except OSError as exc:
            passed = False
            reports.append(f"FAIL {label}: {exc}")
            continue
        out = _tail((result.stdout or "") + (result.stderr or ""))
        if result.returncode != 0:
            passed = False
            reports.append(
                f"FAIL {label}: exit {result.returncode}\n{out}"
            )
        else:
            reports.append(f"PASS {label}\n{out}".rstrip())

    if spec.require_clean_git:
        git = shutil.which("git")
        if git is None:
            passed = False
            reports.append("FAIL require_clean_git: git not found")
        else:
            result = subprocess.run(
                [git, "status", "--porcelain"],
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            dirty = (result.stdout or "").strip()
            if result.returncode != 0:
                passed = False
                reports.append(
                    f"FAIL require_clean_git: git exited {result.returncode}"
                )
            elif dirty:
                passed = False
                reports.append(f"FAIL require_clean_git:\n{_tail(dirty)}")
            else:
                reports.append("PASS require_clean_git")

    return AcceptanceResult(passed=passed, reports=reports)


def _evidence_path(workspace: Path, raw: str) -> str:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    try:
        return str(candidate.resolve())
    except OSError:
        return str(candidate)


def evaluate_closure(
    *,
    state: ProjectState | None,
    task_list: TaskList,
    task_state: TaskStateFile,
    contract_state: ContractStateFile,
    review: TerminalReviewHandoff,
    workspace: Path,
    read_paths: set[str] | None = None,
    acceptance: AcceptanceResult | None = None,
    open_attention: int = 0,
) -> ClosureDecision:
    """All machine checks and the reviewer must agree to close."""
    blockers: list[str] = []

    if not isinstance(state, MissionRunning):
        blockers.append(
            f"state is {getattr(state, 'state', None)!r}, not mission_running"
        )
    if open_attention:
        blockers.append(f"{open_attention} attention item(s) still open")

    for task in task_list.tasks:
        status = task_state.status_of(task.id)
        if task.type == "gate":
            if status != "cleared":
                blockers.append(f"gate {task.id} is {status}")
        elif status not in ("cleared", "superseded"):
            blockers.append(f"task {task.id} is {status}")

    for assertion_id, entry in contract_state.items.items():
        if entry.status != "passed":
            blockers.append(f"assertion {assertion_id} is {entry.status}")

    if not review.done:
        blockers.append("terminal reviewer reported gaps")

    if review.done and read_paths is not None:
        if not review.evidence:
            blockers.append("terminal reviewer done=true without evidence")
        for raw in review.evidence:
            resolved = _evidence_path(workspace, raw)
            if resolved not in read_paths:
                blockers.append(
                    f"reviewer cited {raw!r} but did not read it through ACP"
                )

    if acceptance is not None and not acceptance.passed:
        blockers.append("mission acceptance command failed")
        blockers.extend(
            report for report in acceptance.reports if report.startswith("FAIL")
        )

    return ClosureDecision(allowed=not blockers, blockers=blockers)


__all__ = [
    "AcceptanceResult",
    "ClosureCommand",
    "ClosureDecision",
    "ClosureSpec",
    "closure_spec_path",
    "evaluate_closure",
    "load_closure_spec",
    "run_acceptance",
]
