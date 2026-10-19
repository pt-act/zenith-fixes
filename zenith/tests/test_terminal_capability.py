"""The ACP terminal capability is opt-in, and actually works when enabled.

The audit made agent command execution default-off (C-3): `terminal/create`
cannot be confined to the workspace by this client, so it is refused unless the
operator grants it. That is a *capability* switch, not a removal — a coding
agent that cannot run a build or test command is not much of a coding agent, so
this file pins both halves:

  - with `allow_agent_terminal=True` the whole `terminal/*` round trip works
    (create -> wait_for_exit -> output -> release), driven by a mock adapter
    that behaves like an agent needing to run a command;
  - with it off, the same adapter gets a refusal instead of a silent hang.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from zenith_harness.acp_runner import ACPClient

MOCK_AGENT = Path(__file__).parent / "mock_terminal_acp_agent.py"


async def _run_mission(
    tmp_path: Path,
    *,
    allow_terminal: bool,
    terminal_cwd: str | None = None,
    terminal_cmd: str = "echo",
    terminal_args: list[str] | None = None,
) -> dict:
    """Drive one mock node and return the handoff it produced."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    handoff = tmp_path / "handoff.json"

    env = {
        **__import__("os").environ,
        "ZENITH_HANDOFF_PATH": str(handoff),
        "ZENITH_NODE_ID": "w1",
        "ZENITH_NODE_TYPE": "work",
        "MOCK_TERMINAL_CMD": terminal_cmd,
        "MOCK_TERMINAL_ARGS": json.dumps(terminal_args or ["ZENITH_TERMINAL_OK"]),
    }
    if terminal_cwd:
        env["MOCK_TERMINAL_CWD"] = terminal_cwd

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(MOCK_AGENT),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(workspace),
        env=env,
    )
    client = ACPClient(
        proc,
        str(workspace),
        allow_terminal=allow_terminal,
        permission_policy="deny",
        request_timeout=20.0,
    )
    try:
        await client.start()
        await client.send_request(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": True, "writeTextFile": True},
                    "terminal": allow_terminal,
                },
                "clientInfo": {"name": "test", "version": "0.1.0"},
            },
        )
        session = await client.send_request(
            "session/new", {"cwd": str(workspace), "mcpServers": []}
        )
        await client.send_request(
            "session/prompt",
            {
                "prompt": [{"type": "text", "text": "run the command"}],
                "sessionId": session["sessionId"],
            },
            timeout=20.0,
        )
    finally:
        await client.cleanup()
        if proc.returncode is None:
            proc.terminate()

    assert handoff.exists(), "mock agent produced no handoff"
    return json.loads(handoff.read_text())


@pytest.mark.asyncio
async def test_terminal_round_trip_works_when_enabled(tmp_path: Path) -> None:
    """The opt-in must actually grant a working terminal, not just a flag."""
    data = await _run_mission(tmp_path, allow_terminal=True)
    assert data["done"] is True, data
    assert "ZENITH_TERMINAL_OK" in data["report"], data["report"]


@pytest.mark.asyncio
async def test_terminal_refused_when_disabled(tmp_path: Path) -> None:
    """Default-off: the adapter is told no, and it does not hang."""
    data = await _run_mission(tmp_path, allow_terminal=False)
    assert data["done"] is False, data
    assert "refused" in data["report"].lower(), data["report"]


@pytest.mark.asyncio
async def test_terminal_refusal_is_logged_for_the_operator(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The refusal must be diagnosable: otherwise the node just 'does nothing'."""
    with caplog.at_level("WARNING", logger="zenith_harness.acp_runner"):
        await _run_mission(tmp_path, allow_terminal=False)
    assert any(
        "ZENITH_ALLOW_AGENT_TERMINAL" in rec.getMessage() for rec in caplog.records
    ), [r.getMessage() for r in caplog.records]


@pytest.mark.asyncio
async def test_terminal_cwd_outside_workspace_is_refused(tmp_path: Path) -> None:
    """Granting the terminal does not grant an arbitrary working directory."""
    data = await _run_mission(
        tmp_path, allow_terminal=True, terminal_cwd=str(tmp_path)
    )
    assert data["done"] is False, data
    assert "refused" in data["report"].lower(), data["report"]


@pytest.mark.asyncio
async def test_terminal_args_are_not_shell_expanded(tmp_path: Path) -> None:
    """Arguments reach the child literally — no shell interprets them.

    `echo` prints every argument, so the literal text `$(id)` and the word
    `INJECTED` both appear in the output. What must *not* appear is the result
    of running them: no `uid=` (command substitution) and no second command
    executed.
    """
    data = await _run_mission(
        tmp_path,
        allow_terminal=True,
        terminal_args=["$(id)", ";", "echo INJECTED"],
    )
    assert data["done"] is True, data
    report = data["report"]
    # Literal argument text is echoed...
    assert "$(id)" in report, report
    # ...but never evaluated: a shell would have expanded it.
    assert "uid=" not in report, report
    # The whole argv is echoed by the single command; a shell would have run
    # `echo INJECTED` separately and produced a bare INJECTED line.
    echoed = report.split("terminal output:", 1)[1].strip()
    assert echoed == "$(id) ; echo INJECTED", echoed
