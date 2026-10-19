"""End-to-end smoke: drive a real Zenith mission with Hermes ACP workers.

Mirrors tests/test_smoke_real_acp.py but points the worker/validator/terminal
reviewer at `hermes-acp` (the provider this install was set up for).

Run:
    cd /workspace/home/repos/zenith
    ZENITH_SMOKE_HOME=/workspace/home/.hermes/cache/scratch/zenith-smoke \
    python3 scripts/smoke_hermes_acp.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from zenith_harness.acp_runner import ACPNodeDispatcher, ACPTerminalReviewer  # noqa: E402
from zenith_harness.config import HarnessConfig  # noqa: E402
from zenith_harness.controller import ProjectController  # noqa: E402
from zenith_harness.models import Decision, Task, TaskList  # noqa: E402
from zenith_harness.storage import ProjectStore  # noqa: E402

SMOKE_HOME = Path(os.environ.get("ZENITH_SMOKE_HOME", REPO.parent / ".zenith-smoke"))
WORKSPACE = Path(os.environ.get("ZENITH_SMOKE_WORKSPACE", SMOKE_HOME / "workspace"))
HERMES_ACP = "/workspace/home/.local/bin/hermes-acp"

CONTRACT_BODY = """# VAL-HELLO-001: hello.txt exists

The file `hello.txt` in the workspace root must exist with EXACTLY this content:

```
world
```

(That is the literal six bytes `w`, `o`, `r`, `l`, `d`, `\\\n`.)

No other files modified.
"""

WORKER_BODY = """Use your terminal tool to create `hello.txt` in the workspace
root (cwd) with the exact 6-byte content:

    world

(That is `world` followed by a single newline.)

After verifying the file exists and contains exactly that content, call
`end_node(done=True, report="wrote hello.txt with content 'world'")`
and exit. Do not modify any other files.
"""

VALIDATOR_BODY = """Audit assertion `VAL-HELLO-001`:

1. Read `hello.txt` from the cwd.
2. Check that its content is exactly `world\\n` (6 bytes).
3. Report verdict:
   - if matches: `passed=True`
   - else:       `passed=False` with a note explaining what you observed.

Call `end_node(done=True, report="<what you saw>",
items=[{"item_id": "VAL-HELLO-001", "passed": <bool>}], passed=<bool>)`.
Do not edit any files.
"""


def build_task_list() -> TaskList:
    return TaskList(tasks=[
        Task(id="w1", type="work", body=WORKER_BODY, targets=["VAL-HELLO-001"],
             skill="hello-file-worker"),
        Task(id="v1", type="validate", body=VALIDATOR_BODY, targets=["VAL-HELLO-001"],
             skill="scrutiny-validator", depends_on=["w1"]),
        Task(id="g1", type="gate", body="", targets=["VAL-HELLO-001"], skill=None,
             depends_on=["v1"]),
    ])


def main() -> int:
    bundled = REPO / "src" / "zenith_harness" / "bundled"
    config = HarnessConfig(
        bundled_dir=bundled,
        harness_home=SMOKE_HOME,
        projects_dir=SMOKE_HOME / "projects",
        orchestrator_provider_name="hermes",
        worker_provider_name="hermes",
        worker_acp_command=HERMES_ACP,
        validator_provider_name=None,
        validator_acp_command=None,
        terminal_reviewer_provider_name="hermes",
        terminal_reviewer_acp_command=HERMES_ACP,
        # This smoke mission tells the worker to create a file with its
        # terminal tool, so the run needs the ACP terminal capability. It is
        # off by default (see SECURITY.md), and `HarnessConfig` built directly
        # rather than through `discover()` does not read the environment — so
        # the switch is passed explicitly here.
        allow_agent_terminal=True,
    )

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    for stale in ("hello.txt",):
        p = WORKSPACE / stale
        if p.exists():
            p.unlink()

    store = ProjectStore(config)
    controller = ProjectController(config, ACPNodeDispatcher(config), ACPTerminalReviewer(config))

    env = controller.start_project("smoke: create hello.txt with content 'world'", str(WORKSPACE))
    pid = env.projectId
    print(f"[smoke] project={pid} state={env.state.state}")

    mid = store.load_project(pid).current_mission_id
    assert mid is not None, "no current mission after start_project"
    contract_dir = store.contract_dir(pid, mid)
    contract_dir.mkdir(parents=True, exist_ok=True)
    (contract_dir / "VAL-HELLO-001.md").write_text(CONTRACT_BODY, encoding="utf-8")

    skill_dir = store.zenith_dir(pid) / "skills" / "hello-file-worker"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: hello-file-worker\ndescription: Create the smoke hello file.\n---\n\n"
        "# hello-file-worker\n\nFollow the node body exactly. Create only the requested file.\n",
        encoding="utf-8",
    )

    env = controller.submit_plan(pid, build_task_list())
    print(f"[smoke] after submit_plan state={env.state.state}")

    deadline = time.time() + int(os.environ.get("ZENITH_SMOKE_TIMEOUT_SECONDS", "900"))
    decided_items: set[str] = set()
    requested_closure = False
    while time.time() < deadline:
        env = controller.advance_project(pid)
        print(f"[smoke] state={env.state.state}")
        if env.state.state in ("done", "failed", "aborted"):
            break
        if env.state.state == "attention_needed":
            # A real orchestrator decides each item from evidence; the smoke run
            # auto-decides so it can drive a full mission to closure. Gate
            # checkpoints and clean node reports are `continue`; anything else
            # is reported rather than papered over.
            state_file = store.zenith_runtime_dir(pid) / "state.json"
            items = []
            if state_file.exists():
                import json
                items = (json.loads(state_file.read_text()) or {}).get("items") or []
            decisions = []
            for item in items:
                if item.get("id") in decided_items:
                    continue
                report = str(item.get("report") or "")
                cleared = "cleared: True" in report or "passed: true" in report
                decisions.append(Decision(
                    item_id=item["id"],
                    action="continue" if cleared else "abort",
                    justification=(
                        "smoke: gate/validator evidence supports closure"
                        if cleared else "smoke: non-clean report, aborting mission"
                    ),
                ))
                decided_items.add(item["id"])
            if not decisions:
                print("[smoke] ATTENTION NEEDED with no decidable items")
                return 2
            env = controller.decide_attention(pid, decisions)
            print(f"[smoke] decided {len(decisions)} item(s) -> state={env.state.state}")
            continue
        # Quiescent mission_running: every task cleared. The orchestrator
        # contract is advance_project -> idle -> end_mission, not "advance again
        # until done" — looping here is the smoke script's bug, not Zenith's.
        if not requested_closure:
            requested_closure = True
            print("[smoke] all work cleared; requesting closure via end_mission")
            env = controller.end_mission(pid)
            print(f"[smoke] after end_mission state={env.state.state}")
            if env.state.state in ("done", "failed", "aborted"):
                break
    else:
        print("[smoke] TIMEOUT")
        return 3

    hello = WORKSPACE / "hello.txt"
    content = hello.read_bytes() if hello.exists() else b"<missing>"
    print(f"[smoke] hello.txt exists={hello.exists()} bytes={content!r}")
    ok = content == b"world\n" and env.state.state == "done"
    print("[smoke] RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
