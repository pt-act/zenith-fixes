#!/usr/bin/env python3
"""Mock ACP agent that exercises the client's `terminal/*` surface.

Unlike `mock_acp_agent.py` (which only writes a handoff), this agent behaves
like a coding agent that needs to run a command: on `session/prompt` it asks the
client to create a terminal, waits for it to exit, reads the output, and only
then writes its handoff. That is the flow a worker uses to build or test code,
and it is the flow that breaks if the client stops serving `terminal/create`.

Behaviour is driven by env vars:
  ZENITH_HANDOFF_PATH — where to write the handoff
  ZENITH_NODE_ID      — node id to embed
  ZENITH_NODE_TYPE    — work | validate
  MOCK_TERMINAL_CMD   — command to run (default `echo`)
  MOCK_TERMINAL_ARGS  — JSON list of args (default ["ZENITH_TERMINAL_OK"])
  MOCK_TERMINAL_CWD   — cwd to request; when unset the workspace is used
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_TERMINAL_ID: dict[str, str] = {}


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _write_handoff(report: str, done: bool = True) -> None:
    path = os.environ.get("ZENITH_HANDOFF_PATH")
    if not path:
        return
    node_id = os.environ.get("ZENITH_NODE_ID", "unknown")
    node_type = os.environ.get("ZENITH_NODE_TYPE", "work")
    if node_type == "validate":
        payload = {
            "node_id": node_id,
            "done": done,
            "report": report,
            "items": [{"item_id": "VAL-001", "passed": True}],
            "passed": True,
            "request_attention": False,
        }
    else:
        payload = {
            "node_id": node_id,
            "done": done,
            "report": report,
            "request_attention": False,
        }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(target)


def _request(rid: int, method: str, params: dict) -> None:
    _send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})


def _run_terminal(rid: int, cwd: str | None) -> None:
    """Ask the client for a terminal, run the command, wait, read output."""
    cmd = os.environ.get("MOCK_TERMINAL_CMD", "echo")
    args = json.loads(os.environ.get("MOCK_TERMINAL_ARGS", '["ZENITH_TERMINAL_OK"]'))
    params: dict = {"command": cmd, "args": args}
    if cwd:
        params["cwd"] = cwd
    _request(rid, "terminal/create", params)


def _prompt_flow(req_id: int, session_cwd: str | None) -> None:
    """Run the terminal, then write a handoff derived from what it produced."""
    # The cwd the *agent* requests is independent of the session cwd: an agent
    # can ask for any directory, which is exactly what the client must check.
    requested_cwd = os.environ.get("MOCK_TERMINAL_CWD") or session_cwd
    _run_terminal(1000, requested_cwd)
    # The rest of the flow continues in main() as responses arrive.
    _TERMINAL_ID["pending_prompt_id"] = str(req_id)


def main() -> None:
    pending_terminal: str | None = None
    prompt_id: str | None = None
    terminal_output = ""

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        # ---- responses to our own requests -------------------------------
        if "result" in msg or "error" in msg:
            rid = str(msg.get("id"))
            if rid == "1000":  # terminal/create
                if "error" in msg:
                    _write_handoff(
                        f"terminal/create refused: {msg['error'].get('message')}",
                        done=False,
                    )
                    if prompt_id is not None:
                        _send(
                            {
                                "jsonrpc": "2.0",
                                "id": int(prompt_id),
                                "result": {"stopReason": "end_turn"},
                            }
                        )
                    continue
                pending_terminal = msg["result"]["terminalId"]
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": 1001,
                        "method": "terminal/wait_for_exit",
                        "params": {"terminalId": pending_terminal},
                    }
                )
            elif rid == "1001":  # terminal/wait_for_exit
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": 1002,
                        "method": "terminal/output",
                        "params": {"terminalId": pending_terminal},
                    }
                )
            elif rid == "1002":  # terminal/output
                terminal_output = (msg.get("result") or {}).get("output", "")
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": 1003,
                        "method": "terminal/release",
                        "params": {"terminalId": pending_terminal},
                    }
                )
            elif rid == "1003":  # terminal/release -> finish the prompt
                _write_handoff(f"terminal output: {terminal_output.strip()}")
                if prompt_id is not None:
                    _send(
                        {
                            "jsonrpc": "2.0",
                            "id": int(prompt_id),
                            "result": {"stopReason": "end_turn"},
                        }
                    )
            continue

        # ---- requests from the client ------------------------------------
        method = msg.get("method")
        req_id = msg.get("id")
        if method == "initialize":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": 1,
                        "agentCapabilities": {"loadSession": False},
                        "authMethods": [],
                    },
                }
            )
        elif method == "session/new":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"sessionId": "mock-terminal-session"},
                }
            )
        elif method == "session/set_mode":
            _send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            params = msg.get("params") or {}
            prompt_id = str(req_id)
            _prompt_flow(req_id, params.get("cwd"))
        elif method == "session/cancel":
            _send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        else:
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )


if __name__ == "__main__":
    main()
