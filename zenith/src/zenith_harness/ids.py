"""Identifier validation — the single choke point for path safety.

Every externally supplied identifier in Zenith (project id, mission id, task
id, spawn token, assertion id) eventually becomes a **path component** under
`$ZENITH_HOME`. Most of them arrive from an MCP tool argument, i.e. from an
agent, which makes them untrusted input: `../../..` in a `project_id` would
otherwise let a tool call read or write outside the harness bucket entirely.

This module is deliberately dependency-free so both `config` and `storage` can
import it without creating an import cycle.
"""
from __future__ import annotations

import re

# Deliberately strict: the ids that reach the filesystem are a timestamp, a
# slug, a mission counter, a task id, an assertion id, or a spawn token. None
# of them may contain a path separator, a parent reference, or a leading dot.
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

MAX_COMPONENT_LENGTH = 200


class UnsafeIdentifierError(ValueError):
    """Raised when an externally supplied identifier cannot be used as a path."""


def safe_path_component(value: str, *, kind: str) -> str:
    """Validate `value` as a single, non-traversing path component.

    Rejects empty values, over-long values, anything outside
    `SAFE_COMPONENT_RE` (which excludes `/`, `\\`, `..`, and leading dots), and
    the bare relative names `.` and `..`.
    """
    if not isinstance(value, str) or not value:
        raise UnsafeIdentifierError(f"{kind} must be a non-empty string")
    if len(value) > MAX_COMPONENT_LENGTH:
        raise UnsafeIdentifierError(
            f"{kind} is too long ({len(value)} chars, max {MAX_COMPONENT_LENGTH})"
        )
    if value in {".", ".."}:
        raise UnsafeIdentifierError(f"{kind} {value!r} is a relative path")
    if not SAFE_COMPONENT_RE.fullmatch(value):
        raise UnsafeIdentifierError(
            f"{kind} {value!r} is not a safe path component; it must match "
            f"{SAFE_COMPONENT_RE.pattern}"
        )
    return value


def safe_spawn_ts(value: str) -> str:
    """Validate a spawn timestamp token (also used as a filename component)."""
    return safe_path_component(value, kind="spawn_ts")


def safe_filename(value: str, *, kind: str) -> str:
    """Validate a bare filename component (no separators, no traversal)."""
    return safe_path_component(value, kind=kind)


__all__ = [
    "SAFE_COMPONENT_RE",
    "MAX_COMPONENT_LENGTH",
    "UnsafeIdentifierError",
    "safe_path_component",
    "safe_spawn_ts",
    "safe_filename",
]
