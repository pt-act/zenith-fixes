"""ProjectController — routes the 7 orchestrator MCP tools.

The controller owns:
- envelope construction
- decide_attention validation
- TaskListPatch application via task_list_patch.apply_patch()
- resume-from-disk on every tool call

The coordinator is constructed per-invocation; it has no in-memory state.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import HarnessConfig
from .coordinator import MissionCoordinator
from .dispatcher import NodeDispatcher, TerminalReviewer
from .envelope import EnvelopeDagMode, make_envelope
from .models import (
    AttentionItemInternal,
    AttentionNeeded,
    Aborted,
    ContractStateFile,
    Decision,
    Draft,
    Envelope,
    MissionPlanning,
    MissionRunning,
    ProjectState,
    TaskList,
    TaskListPatch,
    TaskStateFile,
    TaskStatus,
)
from .storage import CorruptArtifact, ProjectBusy, ProjectStore
from .task_list_patch import apply_patch
from .task_validation import (
    ValidationError,
    parse_contract_dir,
    validate_task_list_submission,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolError(Exception):
    code: str
    message: str
    details: list[ValidationError] | None = None

    def __str__(self) -> str:
        if self.details:
            tail = "; ".join(str(d) for d in self.details)
            return f"{self.code}: {self.message} ({tail})"
        return f"{self.code}: {self.message}"


@dataclass
class _DecisionPlan:
    """A fully-validated `decide_attention` batch, staged but not yet written.

    `staged` maps mission id -> (task list, task state, contract ids) with every
    patch already applied in memory; `status_changes` maps mission id -> task id
    -> status for the non-patch actions. `decision_number` is reserved here so
    the audit record cannot be renumbered by a concurrent batch.
    """

    staged: dict[str, tuple[TaskList, TaskStateFile, set[str]]]
    status_changes: dict[str, dict[str, "TaskStatus"]]
    decision_number: int


class ProjectController:
    def __init__(
        self,
        config: HarnessConfig,
        dispatcher: NodeDispatcher,
        terminal_reviewer: TerminalReviewer,
        *,
        store: ProjectStore | None = None,
    ):
        self.config = config
        self.store = store or ProjectStore(config)
        self.dispatcher = dispatcher
        self.terminal_reviewer = terminal_reviewer

    # ------------------------------------------------------------------
    # Tool methods
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self, project_id: str):
        """Serialize a tool call across *processes* (H-04), then recover a
        decision journal left by a crashed commit (H-02)."""
        if not self.store.zenith_runtime_dir(project_id).exists():
            raise ToolError("not_found", f"project {project_id!r} not found")
        try:
            with self.store.project_lock(project_id):
                self.store.recover_decision_journal(project_id)
                yield
        except ProjectBusy as exc:
            raise ToolError("project_busy", str(exc)) from exc

    def start_project(
        self,
        brief: str,
        workspace_dir: str,
        *,
        request_id: str | None = None,
        new_project: bool = False,
    ) -> Envelope:
        if not brief.strip():
            raise ToolError("invalid_brief", "brief is empty")
        resolved = str(Path(workspace_dir).expanduser().resolve())
        # M-06: a retry after a timeout must not create a second project. The
        # default key is deterministic from the brief+workspace; callers that
        # genuinely want another project pass new_project=True (or a fresh
        # request_id).
        key = request_id or hashlib.sha256(
            f"{brief.strip()}\0{resolved}".encode("utf-8")
        ).hexdigest()
        with self.store.global_lock("start"):
            if not new_project:
                existing = self.store.find_project_by_request_id(key)
                if existing is not None:
                    with self.store.project_lock(existing.id):
                        self.store.recover_decision_journal(existing.id)
                        return self._build_envelope(
                            existing.id, dag_mode="none"
                        )
            record = self.store.create_project(
                brief, workspace_dir, request_id=None if new_project else key
            )
            mission_id = self.store.generate_mission_id(1)
            record.current_mission_id = mission_id
            self.store.save_project(record)
            self.store.save_state(
                record.id, MissionPlanning(mission_id=mission_id)
            )
            return self._build_envelope(record.id, dag_mode="none")

    def submit_plan(self, project_id: str, task_list: TaskList) -> Envelope:
        with self._locked(project_id):
            state = self._require_state(project_id)
            if not isinstance(state, MissionPlanning):
                raise ToolError(
                    "wrong_state",
                    f"submit_plan requires MissionPlanning; got {state.state}",
                )
            mid = state.mission_id
            contract_dir = self.store.ensure_contract_dir(project_id, mid)
            ids, parse_errs = parse_contract_dir(contract_dir)
            if parse_errs:
                raise ToolError(
                    "invalid_contract_dir",
                    "contract directory invalid",
                    details=parse_errs,
                )
            errs = validate_task_list_submission(ids, task_list)
            if errs:
                raise ToolError(
                    "invalid_task_list", "task list validation failed", details=errs
                )

            self.store.save_task_list(project_id, mid, task_list)
            task_state = TaskStateFile()
            for task in task_list.tasks:
                task_state.set_status(task.id, "pending")
            self.store.save_task_state(project_id, mid, task_state)
            cs = ContractStateFile()
            for aid in ids:
                cs.items.setdefault(aid, _make_pending())
            self.store.save_contract_state(project_id, mid, cs)
            self.store.save_state(project_id, MissionRunning(mission_id=mid))
            return self._build_envelope(project_id, dag_mode="frontier")

    def advance_project(
        self,
        project_id: str,
        max_steps: int | None = None,
    ) -> Envelope:
        with self._locked(project_id):
            try:
                self.store.sync_workspace_skill_surfaces(project_id)
                coordinator = MissionCoordinator(
                    self.store, project_id, self.dispatcher, self.terminal_reviewer
                )
                steps = 0
                while True:
                    result = coordinator.step()
                    if result.kind in ("attention_needed", "terminal", "idle"):
                        break
                    steps += 1
                    if max_steps is not None and steps >= max_steps:
                        break
                self.store.sync_workspace_skill_surfaces(project_id)
            except CorruptArtifact as exc:
                raise ToolError(
                    "corrupt_artifact",
                    f"{exc}. The project cannot advance until the file is "
                    "repaired or removed; abort_project still works.",
                ) from exc
            return self._build_envelope(project_id, dag_mode="frontier")

    def end_mission(self, project_id: str) -> Envelope:
        with self._locked(project_id):
            state = self._require_state(project_id)
            if not isinstance(state, MissionRunning):
                raise ToolError(
                    "wrong_state",
                    f"end_mission requires MissionRunning; got {state.state}",
                )
            coordinator = MissionCoordinator(
                self.store, project_id, self.dispatcher, self.terminal_reviewer
            )
            try:
                result = coordinator.close_mission(state.mission_id)
            except CorruptArtifact as exc:
                raise ToolError(
                    "corrupt_artifact",
                    f"{exc}. Repair or remove the corrupt mission artifact "
                    "before requesting closure.",
                ) from exc
            if result.kind == "idle":
                raise ToolError(
                    "mission_not_ready_to_close",
                    result.detail or "mission still has runnable task work",
                )
            return self._build_envelope(project_id, dag_mode="none")

    def decide_attention(
        self,
        project_id: str,
        decisions: list[Decision],
    ) -> Envelope:
        with self._locked(project_id):
            state = self._require_state(project_id)
            if not isinstance(state, AttentionNeeded):
                raise ToolError(
                    "wrong_state",
                    f"decide_attention requires AttentionNeeded; got {state.state}",
                )
            try:
                open_items = self.store.load_attention(project_id)
            except CorruptArtifact as exc:
                raise ToolError(
                    "corrupt_attention",
                    f"{exc}. Repair or remove {exc.path}; abort_project still works.",
                ) from exc
            validation_errs = self._validate_decisions(decisions, open_items)
            if validation_errs:
                raise ToolError(
                    "invalid_decisions",
                    "decision validation failed",
                    details=validation_errs,
                )
            item_by_id = {it.id: it for it in open_items}

            # An abort is terminal and dominating: it must not be undone by another
            # decision in the same batch. Previously the loop kept the *last*
            # non-None state, so `[abort, continue]` returned MissionRunning while
            # still writing an "aborted" audit record — the mission kept running
            # after the operator aborted it.
            aborting = [d for d in decisions if d.action == "abort"]

            # Apply every patch to in-memory copies first. `_apply_task_list_patch`
            # validates against the live list, so the whole batch is validated
            # before a single byte is written; a failure therefore leaves disk
            # untouched and the operator can simply retry a corrected batch.
            plan = self._plan_decisions(project_id, decisions, item_by_id)
            paths, directories = self._decision_write_paths(
                project_id, plan, decisions, state
            )
            # H-02: journal the pre-images before the first write. A failure or
            # crash mid-commit is rolled back to this snapshot, so the same
            # decision can be retried instead of hitting task_id_collision.
            self.store.begin_decision_journal(
                project_id, paths, directories=directories
            )
            try:
                self._commit_decision_plan(project_id, plan)

                if aborting:
                    reason = aborting[0].justification or "aborted"
                    record = self.store.load_project(project_id)
                    mid = (
                        self._current_mission_id(record, state)
                        or record.current_mission_id
                    )
                    if mid:
                        try:
                            self.store.seal_mission(
                                project_id, mid, status="aborted", body=reason
                            )
                        except FileNotFoundError:
                            pass
                    next_state: ProjectState = Aborted(reason=reason)
                else:
                    next_state = self._apply_decisions(
                        project_id, decisions, item_by_id
                    )

                # Allocate the record number and write the audit trail under the
                # same call, using the number reserved during planning so a
                # concurrent batch cannot claim it.
                self.store.append_decision_record(
                    project_id,
                    decisions,
                    open_items,
                    decision_number=plan.decision_number,
                )
                self.store.consume_attention(
                    project_id, {d.item_id for d in decisions}
                )
                self.store.save_state(project_id, next_state)
            except CorruptArtifact as exc:
                self.store.rollback_decision_journal(project_id)
                raise ToolError(
                    "corrupt_artifact",
                    f"{exc}. The decision was rolled back; repair or remove "
                    "the corrupt file and retry.",
                ) from exc
            except BaseException:
                self.store.rollback_decision_journal(project_id)
                raise
            else:
                self.store.finish_decision_journal(project_id)
            return self._build_envelope(project_id, dag_mode="none")

    def inspect_project(self, project_id: str) -> Envelope:
        with self._locked(project_id):
            return self._build_envelope(project_id, dag_mode="full")

    def abort_project(self, project_id: str, reason: str) -> Envelope:
        # Abort is the recovery path, so it must not depend on any artifact
        # parsing successfully. A corrupt project.json/state.json still lets
        # us address the bucket and write a terminal Aborted state.
        with self._locked(project_id):
            try:
                record = self.store.load_project(project_id)
            except CorruptArtifact:
                record = None
            try:
                state = self.store.load_state(project_id)
            except CorruptArtifact:
                state = None
            mid = None
            if record is not None:
                mid = self._current_mission_id(record, state)
            elif state is not None and isinstance(
                state, (MissionPlanning, MissionRunning)
            ):
                mid = state.mission_id
            if mid:
                try:
                    self.store.seal_mission(
                        project_id, mid, status="aborted", body=reason
                    )
                except FileNotFoundError:
                    pass
            self.store.clear_attention(project_id)
            self.store.save_state(project_id, Aborted(reason=reason))
            return self._build_envelope(project_id, dag_mode="none")

    # ------------------------------------------------------------------
    # Decision pipeline
    # ------------------------------------------------------------------

    def _validate_decisions(
        self,
        decisions: list[Decision],
        open_items: list[AttentionItemInternal],
    ) -> list[ValidationError]:
        errs: list[ValidationError] = []
        item_ids = {it.id for it in open_items}
        decided_ids = {d.item_id for d in decisions}
        for missing in sorted(item_ids - decided_ids):
            errs.append(ValidationError("unresolved_attention_item", missing))
        for extra in sorted(decided_ids - item_ids):
            errs.append(ValidationError("unknown_attention_item", extra))
        seen: set[str] = set()
        for dec in decisions:
            if dec.item_id in seen:
                errs.append(
                    ValidationError("duplicate_decision", dec.item_id)
                )
            seen.add(dec.item_id)
        item_by_id = {it.id: it for it in open_items}
        for dec in decisions:
            it = item_by_id.get(dec.item_id)
            if it is None:
                continue
            if dec.action == "retry" and it.kind != "node_failed":
                errs.append(
                    ValidationError("invalid_action", "retry is only valid for node_failed")
                )
            if dec.action == "next_mission" and it.kind != "terminal_review":
                errs.append(
                    ValidationError(
                        "invalid_action",
                        "next_mission is only valid for runtime closure reports",
                    )
                )
            if dec.action == "patch":
                if dec.patch is None:
                    errs.append(
                        ValidationError(
                            "missing_patch",
                            f"item {dec.item_id} action=patch requires patch",
                        )
                    )
                elif dec.patch.is_empty:
                    errs.append(
                        ValidationError("empty_patch", dec.item_id)
                    )
        return errs

    def _apply_decisions(
        self,
        project_id: str,
        decisions: list[Decision],
        item_by_id: dict[str, AttentionItemInternal],
    ) -> ProjectState:
        next_state: ProjectState | None = None
        for dec in decisions:
            item = item_by_id[dec.item_id]
            outcome = self._apply_one(project_id, item, dec)
            if outcome is not None:
                next_state = outcome
        if next_state is not None:
            return next_state
        record = self.store.load_project(project_id)
        mid = record.current_mission_id
        if mid:
            return MissionRunning(mission_id=mid)
        return Draft()

    def _apply_one(
        self,
        project_id: str,
        item: AttentionItemInternal,
        decision: Decision,
    ) -> ProjectState | None:
        """Compute the resulting state for one decision.

        The *disk effects* of every decision are already applied by
        `_commit_decision_plan`; this only decides which state the batch ends
        in. `patch` deliberately returns MissionRunning here (the patch was
        already staged) so the state resolution stays a pure function of the
        decision kinds.
        """
        mid = item.mission_id
        kind = item.kind
        action = decision.action

        if action == "patch":
            return MissionRunning(mission_id=mid)

        if action == "retry" and item.node_id:
            return MissionRunning(mission_id=mid)

        if action == "continue":
            if kind in ("node_failed", "gate_failed") and item.node_id:
                return MissionRunning(mission_id=mid)
            if kind == "terminal_review":
                return None
            return MissionRunning(mission_id=mid)

        if action == "abort":
            return Aborted(reason=decision.justification or "aborted")

        if action == "next_mission":
            return self._seal_and_plan_next(project_id, mid)

        return None

    def _seal_and_plan_next(self, project_id: str, mid: str) -> ProjectState:
        self.store.seal_mission(
            project_id,
            mid,
            status="done_with_acknowledged_gaps",
            body="Sealed with gap(s) acknowledged via next_mission decision.",
        )
        existing = self.store.list_missions(project_id)
        new_mid = self.store.generate_mission_id(len(existing) + 1)
        record = self.store.load_project(project_id)
        record.current_mission_id = new_mid
        self.store.save_project(record)
        return MissionPlanning(mission_id=new_mid)

    def _apply_task_list_patch(
        self, project_id: str, mid: str, patch: TaskListPatch
    ) -> None:
        tl = self.store.load_task_list(project_id, mid)
        task_state = self.store.load_task_state(project_id, mid)
        contract_state = self.store.load_contract_state(project_id, mid)
        old_ids = set(contract_state.items.keys())
        all_ids_on_disk = set(self.store.list_contract_assertions(project_id, mid))
        new_on_disk = all_ids_on_disk - old_ids

        patched_tl, patched_state, patched_ids, errs = apply_patch(
            tl,
            task_state,
            old_ids,
            patch,
            new_contract_ids_on_disk=new_on_disk,
        )
        if errs:
            raise ToolError(
                "invalid_patch", "patch validation failed", details=errs
            )
        self.store.save_task_list(project_id, mid, patched_tl)
        self.store.save_task_state(project_id, mid, patched_state)
        cs = self.store.load_contract_state(project_id, mid)
        for aid in patched_ids - old_ids:
            cs.items.setdefault(aid, _make_pending())
        self.store.save_contract_state(project_id, mid, cs)

    # ------------------------------------------------------------------
    # Decision planning (validate everything before writing anything)
    # ------------------------------------------------------------------

    def _plan_decisions(
        self,
        project_id: str,
        decisions: list[Decision],
        item_by_id: dict[str, AttentionItemInternal],
    ) -> _DecisionPlan:
        """Validate and stage every decision in memory, writing nothing.

        The runtime's own contract is "one `decide_attention` call resolves the
        whole open set". Applying the decisions one-by-one to disk made that
        non-atomic: if a later patch failed validation the earlier ones were
        already persisted, the attention items were left open, no audit record
        was written, and retrying the *same* batch then failed forever on
        `task_id_collision` because the first patch had already added its task.
        The only recovery was aborting the project.

        Planning first means a rejected batch leaves disk byte-identical, so the
        operator can correct and retry.
        """
        staged: dict[str, tuple[TaskList, TaskStateFile, set[str]]] = {}
        decision_number = self.store.next_decision_number(project_id)

        for dec in decisions:
            item = item_by_id.get(dec.item_id)
            if item is None:
                continue
            mid = item.mission_id
            if dec.action == "patch":
                assert dec.patch is not None
                base = staged.get(
                    mid, self._load_patch_base(project_id, mid)
                )
                staged[mid] = self._stage_patch(project_id, mid, base, dec.patch)

        # Compute the resulting task-state view for each mission, folding in the
        # status changes the non-patch actions imply, so the final state
        # decision matches what will be on disk.
        status_changes: dict[str, dict[str, TaskStatus]] = {}
        for dec in decisions:
            item = item_by_id.get(dec.item_id)
            if item is None or dec.action == "patch":
                continue
            mid = item.mission_id
            node_id = item.node_id
            if not node_id:
                continue
            changes = status_changes.setdefault(mid, {})
            if dec.action == "retry":
                changes[node_id] = "pending"
            elif dec.action == "continue" and item.kind in ("node_failed", "gate_failed"):
                changes[node_id] = "cleared"

        return _DecisionPlan(
            staged=staged,
            status_changes=status_changes,
            decision_number=decision_number,
        )

    def _load_patch_base(
        self, project_id: str, mid: str
    ) -> tuple[TaskList, TaskStateFile, set[str]]:
        tl = self.store.load_task_list(project_id, mid)
        task_state = self.store.load_task_state(project_id, mid)
        contract_state = self.store.load_contract_state(project_id, mid)
        return tl, task_state, set(contract_state.items.keys())

    def _stage_patch(
        self,
        project_id: str,
        mid: str,
        base: tuple[TaskList, TaskStateFile, set[str]],
        patch: TaskListPatch,
    ) -> tuple[TaskList, TaskStateFile, set[str]]:
        """Apply one patch on top of already-staged state (no disk writes)."""
        tl, task_state, old_ids = base
        all_ids_on_disk = set(self.store.list_contract_assertions(project_id, mid))
        new_on_disk = all_ids_on_disk - old_ids
        patched_tl, patched_state, patched_ids, errs = apply_patch(
            tl,
            task_state,
            old_ids,
            patch,
            new_contract_ids_on_disk=new_on_disk,
        )
        if errs:
            raise ToolError("invalid_patch", "patch validation failed", details=errs)
        return patched_tl, patched_state, patched_ids

    def _decision_write_paths(
        self,
        project_id: str,
        plan: "_DecisionPlan",
        decisions: list[Decision],
        state: ProjectState,
    ) -> tuple[list[Path], list[Path]]:
        """Every file a decide_attention batch may write, for the journal."""
        runtime = self.store.zenith_runtime_dir(project_id)
        paths: set[Path] = {
            runtime / "attention.json",
            runtime / "state.json",
            runtime / "project.json",
        }
        for mid in plan.staged:
            mdir = self.store.mission_runtime_dir(project_id, mid)
            paths.update(
                {
                    mdir / "tasks.json",
                    mdir / "task-state.json",
                    mdir / "contract-state.json",
                }
            )
        for mid in plan.status_changes:
            if mid not in plan.staged:
                paths.add(self.store.mission_runtime_dir(project_id, mid) / "task-state.json")

        if any(d.action in ("abort", "next_mission") for d in decisions):
            try:
                record = self.store.load_project(project_id)
            except CorruptArtifact:
                record = None
            closeout_mid: str | None = None
            if record is not None:
                closeout_mid = self._current_mission_id(record, state)
            elif isinstance(state, (MissionPlanning, MissionRunning)):
                closeout_mid = state.mission_id
            if closeout_mid:
                paths.add(
                    self.store.mission_dir(project_id, closeout_mid) / "closeout.md"
                )

        return sorted(paths), [self.store.decisions_dir(project_id)]

    def _commit_decision_plan(self, project_id: str, plan: "_DecisionPlan") -> None:
        """Persist a fully-validated plan. Only called once planning succeeds."""
        for mid, changes in plan.status_changes.items():
            if mid in plan.staged:
                # Fold the non-patch status changes into the staged task state
                # so both land in the same single write.
                _tl, staged_state, contract_ids = plan.staged[mid]
                for node_id, status in changes.items():
                    staged_state.set_status(node_id, status)
                plan.staged[mid] = (_tl, staged_state, contract_ids)
            else:
                task_state = self.store.load_task_state(project_id, mid)
                for node_id, status in changes.items():
                    task_state.set_status(node_id, status)
                self.store.save_task_state(project_id, mid, task_state)

        for mid, (tl, task_state, contract_ids) in plan.staged.items():
            cs = self.store.load_contract_state(project_id, mid)
            for aid in contract_ids:
                cs.items.setdefault(aid, _make_pending())
            self.store.save_task_list(project_id, mid, tl)
            self.store.save_task_state(project_id, mid, task_state)
            self.store.save_contract_state(project_id, mid, cs)

    # ------------------------------------------------------------------
    # Envelope
    # ------------------------------------------------------------------

    def _build_envelope(
        self, project_id: str, *, dag_mode: EnvelopeDagMode = "summary"
    ) -> Envelope:
        try:
            record = self.store.load_project(project_id)
        except CorruptArtifact:
            # abort_project must be able to return an envelope even when the
            # project record is damaged; paths are derivable from the id.
            record = None
        try:
            state = self.store.load_state(project_id) or Draft()
        except CorruptArtifact as exc:
            raise ToolError(
                "corrupt_state",
                f"{exc}. Repair or remove {exc.path} (abort_project can still "
                "stop the project).",
            ) from exc
        tl: TaskList | None = None
        task_state: TaskStateFile | None = None
        mid = self._current_mission_id(record, state)
        if mid:
            try:
                tl = self.store.load_task_list(project_id, mid)
                task_state = self.store.load_task_state(project_id, mid)
            except (FileNotFoundError, CorruptArtifact):
                # A damaged task list must not hide the rest of the envelope;
                # advance/decide surface the CorruptArtifact as a ToolError.
                pass
        project_root = str(self.store.zenith_dir(project_id))
        harness_root = str(self.store.bucket_root(project_id))
        runtime_root = str(self.store.zenith_runtime_dir(project_id))
        return make_envelope(
            project_id,
            state,
            project_root,
            harness_root,
            tl,
            task_state,
            dag_mode=dag_mode,
            runtime_root=runtime_root,
            mission_id=mid,
        )

    @staticmethod
    def _current_mission_id(record, state) -> str | None:
        if isinstance(state, (MissionPlanning, MissionRunning)):
            return state.mission_id
        if record is not None:
            return record.current_mission_id
        return None

    def _require_state(self, project_id: str) -> ProjectState:
        try:
            state = self.store.load_state(project_id)
        except CorruptArtifact as exc:
            raise ToolError(
                "corrupt_state",
                f"{exc}. Repair or remove {exc.path}; abort_project still works.",
            ) from exc
        if state is None:
            raise ToolError("not_found", f"project {project_id!r} has no state")
        return state


def build_controller(config: HarnessConfig) -> ProjectController:
    """Construct the controller used by the MCP server AND the CLI (M-9).

    `server.main` used to wire the ACP dispatcher while `zenith abort-project`
    built its own controller around MockDispatcher/MockTerminalReviewer, so the
    two surfaces could drift. Both now go through this factory; the ACP import
    failure path falls back to no-op collaborators in exactly one place.
    """
    try:
        from .acp_runner import ACPNodeDispatcher, ACPTerminalReviewer

        dispatcher: NodeDispatcher = ACPNodeDispatcher(config)
        reviewer: TerminalReviewer = ACPTerminalReviewer(config)
    except Exception as exc:  # pragma: no cover — diagnostic
        logger.warning("Falling back to no-op dispatcher: %s", exc)
        from .dispatcher import MockDispatcher, MockTerminalReviewer
        from .models import TerminalReviewHandoff, WorkHandoff

        dispatcher = MockDispatcher(
            lambda r: WorkHandoff(
                node_id=r.task.id, done=False, report="no ACP runtime available"
            )
        )
        reviewer = MockTerminalReviewer(
            TerminalReviewHandoff(done=True, report="")
        )
    return ProjectController(config, dispatcher, reviewer)


def _make_pending():
    from .models import ContractStateEntry
    return ContractStateEntry()


__all__ = ["ProjectController", "ToolError", "build_controller"]
