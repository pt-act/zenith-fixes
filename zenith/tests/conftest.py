"""Shared test fixtures for v5."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def write_acceptance():
    """Write a minimal passing closure.json for a mission.

    The acceptance command must be portable. `/bin/true` does not exist on
    macOS, so the suite failed there for a reason unrelated to the code under
    test; the running interpreter is always present.
    """

    def _write(store, project_id: str, mission_id: str = "mission-001") -> None:
        path = store.mission_dir(project_id, mission_id) / "closure.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"commands": [{"argv": [sys.executable, "-c", "pass"]}]}
            ),
            encoding="utf-8",
        )

    return _write


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def harness_home(tmp_path: Path) -> Path:
    home = tmp_path / ".zenith"
    home.mkdir()
    return home


@pytest.fixture
def contract_dir(workspace: Path) -> Path:
    d = workspace / ".zenith" / "missions" / "mission-001" / "contract"
    d.mkdir(parents=True)
    return d
