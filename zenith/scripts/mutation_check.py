"""Mutation-test the audit fixes.

Run from the checkout:
    python scripts/mutation_check.py

For each finding: revert the fix in a scratch copy, run the test that is
supposed to pin it, and require the test to FAIL. A test that still passes with
the fix removed is vacuous and proves nothing.

Usage: python mutate_check.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SRC_TREE = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("MUTATION_WORK", "/tmp/zenith-mutation"))
PY = sys.executable


class Mutation:
    def __init__(self, name: str, path: str, old: str, new: str, test: str):
        self.name = name
        self.path = path
        self.old = old
        self.new = new
        self.test = test


MUTATIONS = [
    Mutation(
        "C-2 workspace containment removed",
        "src/zenith_harness/acp_runner.py",
        """        if resolved != self._workspace_root and not resolved.is_relative_to(
            self._workspace_root
        ):
            raise ACPError(
                f"path {raw!r} resolves outside the workspace and is refused"
            )""",
        "        pass",
        "tests/test_audit_regressions.py::test_fs_read_refuses_parent_traversal",
    ),
    Mutation(
        "C-3 terminal allowed by default",
        "src/zenith_harness/acp_runner.py",
        "        if not self._allow_terminal:",
        "        if False:",
        "tests/test_audit_regressions.py::test_terminal_disabled_by_default",
    ),
    Mutation(
        "C-4 permission allowed by default",
        "src/zenith_harness/acp_runner.py",
        '        else:  # "deny"\n            chosen = self._pick_option(options, allow=False)',
        '        else:  # "deny"\n            chosen = self._pick_option(options, allow=True)',
        "tests/test_audit_regressions.py::test_permission_denied_by_default",
    ),
    Mutation(
        "C-6 decide_attention made non-atomic again",
        "src/zenith_harness/controller.py",
        """            plan = self._plan_decisions(project_id, decisions, item_by_id)""",
        """            for _d in decisions:
                _item = item_by_id.get(_d.item_id)
                if _d.action == "patch" and _d.patch is not None and _item is not None:
                    self._apply_task_list_patch(project_id, _item.mission_id, _d.patch)
            plan = _DecisionPlan(staged={}, status_changes={}, decision_number=self.store.next_decision_number(project_id))""",
        "tests/test_audit_regressions.py::test_decide_attention_is_atomic_on_invalid_patch",
    ),
    Mutation(
        "H-1 abort no longer dominates",
        "src/zenith_harness/controller.py",
        "        aborting = [d for d in decisions if d.action == \"abort\"]",
        "        aborting = []",
        "tests/test_audit_regressions.py::test_abort_dominates_other_decisions",
    ),
    Mutation(
        "C-8 corrupt handoff raises again",
        "src/zenith_harness/storage.py",
        """        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            quarantine_corrupt(path, reason=f"unparseable handoff: {exc}")
            return _corrupt_handoff(inferred_node, task_type, f"corrupt handoff file ({exc})")""",
        """        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise""",
        "tests/test_audit_regressions.py::test_corrupt_attempt_raises_attention_not_exception",
    ),
    Mutation(
        "H-2 gate accepts unassigned votes again",
        "src/zenith_harness/coordinator.py",
        "                if item.item_id in expected:",
        "                if item.item_id in gate.targets:",
        "tests/test_audit_regressions.py::test_gate_ignores_verdicts_for_unassigned_targets",
    ),
    Mutation(
        "C-7 project_id not validated",
        "src/zenith_harness/ids.py",
        "    if not SAFE_COMPONENT_RE.fullmatch(value):",
        "    if False:",
        "tests/test_audit_regressions.py::test_unsafe_project_ids_rejected",
    ),
    Mutation(
        "C-7 project ids collide again",
        "src/zenith_harness/storage.py",
        '        return f"{ts}-{slugify(first_line, \'project\')}-{secrets.token_hex(3)}"',
        '        return f"{ts}-{slugify(first_line, \'project\')}"',
        "tests/test_audit_regressions.py::test_generate_project_id_is_collision_resistant",
    ),
    Mutation(
        "H-6 oversized line kills the reader again (guard + finally reverted)",
        "src/zenith_harness/acp_runner.py",
        """        except (BrokenPipeError, ConnectionResetError, OSError, ValueError) as exc:
            # `ValueError` is what StreamReader.readline() raises when a line
            # exceeds the stream limit (asyncio.LimitOverrunError is a
            # subclass). It used to escape this loop, kill the reader task, and
            # leave every pending request — and therefore the whole node —
            # waiting forever.
            logger.warning("ACP read loop terminating: %s", exc)
        finally:
            was_closing = self._closing
            self._closing = True""",
        """        except (BrokenPipeError, ConnectionResetError, OSError, ValueError) as exc:
            raise
        finally:
            pass""",
        "tests/test_audit_regressions.py::test_oversized_line_fails_pending_request_instead_of_hanging",
    ),
    Mutation(
        "M-5 fences not adaptive again",
        "src/zenith_harness/storage.py",
        "    longest = max((len(run) for run in _FENCE_RUN_RE.findall(content)), default=0)\n    marker = \"`\" * max(3, longest + 1)",
        '    marker = "```"',
        "tests/test_audit_regressions.py::test_decision_record_cannot_be_escaped_by_report_text",
    ),
    Mutation(
        "M-3 output limit not clamped",
        "src/zenith_harness/acp_runner.py",
        "            output_limit = min(requested, self._max_terminal_output_bytes)",
        "            output_limit = requested",
        "tests/test_audit_regressions.py::test_terminal_output_limit_is_clamped",
    ),
    Mutation(
        "C-5 handoff node_id not cross-checked",
        "src/zenith_harness/acp_runner.py",
        "        if isinstance(reported, str) and reported != task.id:",
        "        if False:",
        "tests/test_audit_regressions.py::test_handoff_for_a_different_task_is_rejected",
    ),
    Mutation(
        "H-3 brief inlined into reviewer prompt again",
        "src/zenith_harness/acp_runner.py",
        '                "user_request": (\n                    "The original user request is stored verbatim at "',
        '                "user_request": (\n                    brief_text + "\\nThe original user request is stored verbatim at "',
        "tests/test_audit_regressions.py::test_reviewer_prompt_does_not_inline_the_brief",
    ),
    Mutation(
        "H-9 ACP launch goes back through a shell",
        "src/zenith_harness/acp_runner.py",
        """    return await asyncio.create_subprocess_exec(
        *argv,""",
        """    return await asyncio.create_subprocess_shell(
        command,""",
        "tests/test_audit_gaps.py::test_acp_spawn_does_not_interpret_shell_metacharacters",
    ),
    Mutation(
        "M-1 attention queue is uncapped again",
        "src/zenith_harness/attention.py",
        "    max_items: int = MAX_OPEN_ATTENTION_ITEMS,",
        "    max_items: int = 10**9,",
        "tests/test_audit_gaps.py::test_attention_queue_is_capped_and_keeps_failures",
    ),
    Mutation(
        "H-5 workspace shims written unconditionally again",
        "src/zenith_harness/storage.py",
        """        if self.config.workspace_shims:
            self._ensure_symlink_shims(ws, zenith)""",
        """        self._ensure_symlink_shims(ws, zenith)""",
        "tests/test_audit_gaps.py::test_workspace_shims_are_opt_in",
    ),
    Mutation(
        "C-01 ACP agent inherits os.environ again",
        "src/zenith_harness/acp_runner.py",
        """    env = {
        key: os.environ[key]
        for key in _acp_env_allowlist(name)
        if key in os.environ
    }
    env.setdefault("PATH", os.defpath)""",
        """    env = os.environ.copy()""",
        "tests/test_audit_round2.py::test_acp_env_does_not_inherit_unrelated_secrets",
    ),
    Mutation(
        "H-02 decision journal rollback removed",
        "src/zenith_harness/controller.py",
        """            except BaseException:
                self.store.rollback_decision_journal(project_id)
                raise""",
        """            except BaseException:
                raise""",
        "tests/test_audit_round2.py::test_decide_attention_rolls_back_on_io_failure",
    ),
    Mutation(
        "M-01 corrupt state bypasses the guard",
        "src/zenith_harness/storage.py",
        """        return self._load_model(
            path, TypeAdapter(ProjectState), kind="state.json"
        )""",
        """        return TypeAdapter(ProjectState).validate_json(
            path.read_text(encoding="utf-8")
        )""",
        "tests/test_audit_round2.py::test_corrupt_state_is_tool_error_and_abort_works",
    ),
    Mutation(
        "M-10 attention cap drops actionable failures again",
        "src/zenith_harness/attention.py",
        """        if item.kind != "gate_checkpoint" or item.id in kept_ids""",
        """        if item.id in kept_ids""",
        "tests/test_audit_round2.py::test_attention_cap_keeps_actionable_items",
    ),
    Mutation(
        "L-01 Windows splitter falls back to shlex(posix=False)",
        "src/zenith_harness/acp_runner.py",
        """    return _split_windows_command(command)""",
        """    return shlex.split(command, posix=False)""",
        "tests/test_audit_round2.py::test_windows_command_split_preserves_backslashes",
    ),
    Mutation(
        "M-09 agent report cap removed",
        "src/zenith_harness/models.py",
        """    if isinstance(value, str) and len(value) > MAX_AGENT_REPORT_CHARS:
        return value[: MAX_AGENT_REPORT_CHARS - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER
    return value""",
        """    return value""",
        "tests/test_audit_round2.py::test_agent_report_is_truncated",
    ),
    Mutation(
        "closure accepts missing acceptance result",
        "src/zenith_harness/closure.py",
        """    if acceptance is None:
        blockers.append(
            "no closure.json acceptance commands defined; closure is not allowed "
            "without a machine-checkable acceptance result"
        )
    elif not acceptance.passed:""",
        """    if False:
        pass
    elif not acceptance.passed:""",
        "tests/test_closure.py::test_missing_acceptance_result_blocks_closure",
    ),
    Mutation(
        "closure allows an empty acceptance command list",
        "src/zenith_harness/closure.py",
        """    commands: list[ClosureCommand] = Field(min_length=1)""",
        """    commands: list[ClosureCommand] = Field(default_factory=list)""",
        "tests/test_closure.py::test_closure_spec_requires_at_least_one_command",
    ),
]


def run() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    # Copy only the source tree: a .venv or git dir inside the checkout would
    # make each run slow and can shadow the PYTHONPATH we set below.
    shutil.copytree(
        SRC_TREE,
        WORK,
        ignore=shutil.ignore_patterns(
            ".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"
        ),
    )

    results: list[tuple[str, str, str]] = []
    for mut in MUTATIONS:
        target = WORK / mut.path
        original = target.read_text()
        if mut.old not in original:
            results.append((mut.name, "SKIP", "anchor text not found"))
            continue
        target.write_text(original.replace(mut.old, mut.new, 1))
        proc = subprocess.run(
            [PY, "-m", "pytest", mut.test, "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=WORK,
            capture_output=True,
            text=True,
            env={
                "PYTHONPATH": str(WORK / "src"),
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": os.environ.get("HOME", str(Path.home())),
            },
            timeout=300,
        )
        target.write_text(original)  # restore for the next mutation

        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            verdict = "GOOD (test caught it)"
        else:
            verdict = "VACUOUS (test still passed!)"
        # Show which test was affected.
        m = re.search(r"(\d+ (?:failed|passed|error))", out)
        results.append((mut.name, verdict, m.group(1) if m else "no summary"))

    print(f"{'mutation':52s} {'verdict':26s} summary")
    print("-" * 100)
    bad = 0
    for name, verdict, summary in results:
        print(f"{name:52s} {verdict:26s} {summary}")
        if verdict.startswith("VACUOUS") or verdict == "SKIP":
            bad += 1
    print()
    print(f"{len(results) - bad}/{len(results)} mutations were caught by their test")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(run())
