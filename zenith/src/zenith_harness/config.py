"""v5 HarnessConfig.

Layout: `$ZENITH_HOME/projects/<pid>/.zenith` (durable record) and
`.../.zenith-runtime` (orchestrator-only cursors). Ids that become path
components are validated by `ids.safe_path_component`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .providers import (
    ProviderSelection,
    default_worker_provider_name,
    get_provider,
)
from .ids import safe_path_component

DEFAULT_MAX_PARALLEL_NODES = 4

# codex-acp `model_reasoning_effort` values. Also a safety allowlist: the
# resolved value is spliced into a shell command line by acp_runner. Codex's
# "ultra" is deliberately excluded: it is not a reasoning tier (codex
# downgrades the request to "max" on the wire) but a switch to proactive
# multi-agent mode — a lane spawning its own agent swarm inside a harness
# that already orchestrates and validates per-lane work.
VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max")

# How the client answers ACP `session/request_permission`.
#
#   "deny"       — refuse every request (default). The agent cannot take an
#                  action that requires approval, so the harness never silently
#                  authorizes a destructive operation on the operator's behalf.
#   "allow_all"  — approve everything. This was the previous unconditional
#                  behaviour and is required for fully unattended runs; it is
#                  now an explicit operator opt-in.
#   "allow_read" — approve only requests whose offered options are read-only
#                  kinds, deny the rest.
VALID_PERMISSION_POLICIES = ("deny", "allow_all", "allow_read")

DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_PROMPT_TIMEOUT = 3600.0
DEFAULT_MAX_TERMINAL_OUTPUT_BYTES = 8 * 1024 * 1024


def _bundled_dir() -> Path:
    return (Path(__file__).resolve().parent / "bundled").resolve()


def _resolve_optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _resolve_max_parallel(value: str | None) -> int:
    if not value:
        return DEFAULT_MAX_PARALLEL_NODES
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_MAX_PARALLEL_NODES
    return max(1, parsed)


def _resolve_reasoning_effort(value: str | None, *, env_var: str) -> str | None:
    """None passes through (provider default); anything else must be on the
    allowlist — a typo silently ignored would spend xhigh the user thought
    they had dialed down."""
    if not value:
        return None
    if value not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"{env_var}={value!r} is not a valid reasoning effort; "
            f"choose one of: {', '.join(VALID_REASONING_EFFORTS)}"
        )
    return value


def validate_reasoning_effort(value: str | None, *, source: str) -> str | None:
    """Validate a reasoning effort at the point of use.

    `discover()` validates the env var, but the value is also spliced into a
    shell command line by `_augment_acp_command`, so it is re-checked there:
    a code path that constructs a config without `discover()` (tests, an
    embedding host) must not be able to inject arbitrary text into argv.
    """
    if value is None:
        return None
    if value not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"{source}: {value!r} is not a valid reasoning effort; "
            f"choose one of: {', '.join(VALID_REASONING_EFFORTS)}"
        )
    return value


def _resolve_bool(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_permission_policy(value: str | None) -> str:
    if not value:
        return "deny"
    if value not in VALID_PERMISSION_POLICIES:
        raise ValueError(
            f"ZENITH_ACP_PERMISSION_POLICY={value!r} is not valid; "
            f"choose one of: {', '.join(VALID_PERMISSION_POLICIES)}"
        )
    return value


def _resolve_max_bytes(value: str | None, *, default: int, env_var: str) -> int:
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{env_var}={value!r} is not an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{env_var}={value!r} must be positive")
    return parsed


def _resolve_positive_float(value: str | None, *, default: float, env_var: str) -> float:
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{env_var}={value!r} is not a number") from exc
    if parsed <= 0:
        raise ValueError(f"{env_var}={value!r} must be positive")
    return parsed


@dataclass(frozen=True)
class HarnessConfig:
    """Static configuration loaded from env. Per-call overrides allowed via `with_*`."""

    bundled_dir: Path
    harness_home: Path  # ZENITH_HOME (default ~/.zenith)
    projects_dir: Path  # ZENITH_PROJECTS_DIR (default <harness_home>/projects)
    orchestrator_provider_name: str
    worker_provider_name: str
    worker_acp_command: str | None
    validator_provider_name: str | None
    validator_acp_command: str | None
    terminal_reviewer_provider_name: str | None
    terminal_reviewer_acp_command: str | None
    max_parallel_nodes: int = DEFAULT_MAX_PARALLEL_NODES
    # Per-role reasoning effort for providers whose ACP command accepts one
    # (codex today). None means the provider default ("xhigh" for codex).
    worker_reasoning_effort: str | None = None
    validator_reasoning_effort: str | None = None
    terminal_reviewer_reasoning_effort: str | None = None
    # Safety/robustness controls (see VALID_PERMISSION_POLICIES).
    permission_policy: str = "deny"
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    prompt_timeout: float = DEFAULT_PROMPT_TIMEOUT
    max_terminal_output_bytes: int = DEFAULT_MAX_TERMINAL_OUTPUT_BYTES
    # ACP terminal access (agent-driven command execution) is off unless the
    # operator opts in. It cannot be safely confined to the workspace with the
    # current client implementation, so it is not granted by default.
    allow_agent_terminal: bool = False
    # Sandboxed vs unrestricted agent execution. Unrestricted (no sandbox, no
    # approval) is the historical behaviour and stays available, but it is now
    # an explicit opt-in instead of an unmarked default.
    allow_unsandboxed: bool = False

    @classmethod
    def discover(cls) -> HarnessConfig:
        if os.environ.get("ZENITH_PROJECT_BUCKET_DIR"):
            raise RuntimeError(
                "ZENITH_PROJECT_BUCKET_DIR was removed in memory_v2; the project "
                "bucket is always $ZENITH_HOME/projects/<pid>/. Unset the env var."
            )
        harness_home = (
            Path(os.environ.get("ZENITH_HOME") or (Path.home() / ".zenith"))
            .expanduser()
            .resolve()
        )
        projects_dir = (
            _resolve_optional_path(os.environ.get("ZENITH_PROJECTS_DIR"))
            or harness_home / "projects"
        )
        orchestrator_provider_name = os.environ.get(
            "ZENITH_ORCHESTRATOR_PROVIDER", "claude"
        )
        worker_provider_name = os.environ.get(
            "ZENITH_WORKER_PROVIDER"
        ) or default_worker_provider_name(orchestrator_provider_name)
        worker_acp_command = os.environ.get("ZENITH_WORKER_ACP_COMMAND")
        validator_provider_name = os.environ.get("ZENITH_VALIDATOR_PROVIDER")
        validator_acp_command = os.environ.get("ZENITH_VALIDATOR_ACP_COMMAND")
        terminal_reviewer_provider_name = os.environ.get(
            "ZENITH_TERMINAL_REVIEWER_PROVIDER"
        )
        terminal_reviewer_acp_command = os.environ.get(
            "ZENITH_TERMINAL_REVIEWER_ACP_COMMAND"
        )
        return cls(
            bundled_dir=_bundled_dir(),
            harness_home=harness_home,
            projects_dir=projects_dir,
            orchestrator_provider_name=orchestrator_provider_name,
            worker_provider_name=worker_provider_name,
            worker_acp_command=worker_acp_command,
            validator_provider_name=validator_provider_name,
            validator_acp_command=validator_acp_command,
            terminal_reviewer_provider_name=terminal_reviewer_provider_name,
            terminal_reviewer_acp_command=terminal_reviewer_acp_command,
            max_parallel_nodes=_resolve_max_parallel(
                os.environ.get("ZENITH_MAX_PARALLEL_NODES")
            ),
            worker_reasoning_effort=_resolve_reasoning_effort(
                os.environ.get("ZENITH_WORKER_REASONING_EFFORT"),
                env_var="ZENITH_WORKER_REASONING_EFFORT",
            ),
            validator_reasoning_effort=_resolve_reasoning_effort(
                os.environ.get("ZENITH_VALIDATOR_REASONING_EFFORT"),
                env_var="ZENITH_VALIDATOR_REASONING_EFFORT",
            ),
            terminal_reviewer_reasoning_effort=_resolve_reasoning_effort(
                os.environ.get("ZENITH_TERMINAL_REVIEWER_REASONING_EFFORT"),
                env_var="ZENITH_TERMINAL_REVIEWER_REASONING_EFFORT",
            ),
            permission_policy=_resolve_permission_policy(
                os.environ.get("ZENITH_ACP_PERMISSION_POLICY")
            ),
            request_timeout=_resolve_positive_float(
                os.environ.get("ZENITH_ACP_REQUEST_TIMEOUT"),
                default=DEFAULT_REQUEST_TIMEOUT,
                env_var="ZENITH_ACP_REQUEST_TIMEOUT",
            ),
            prompt_timeout=_resolve_positive_float(
                os.environ.get("ZENITH_ACP_PROMPT_TIMEOUT"),
                default=DEFAULT_PROMPT_TIMEOUT,
                env_var="ZENITH_ACP_PROMPT_TIMEOUT",
            ),
            max_terminal_output_bytes=_resolve_max_bytes(
                os.environ.get("ZENITH_MAX_TERMINAL_OUTPUT_BYTES"),
                default=DEFAULT_MAX_TERMINAL_OUTPUT_BYTES,
                env_var="ZENITH_MAX_TERMINAL_OUTPUT_BYTES",
            ),
            allow_agent_terminal=_resolve_bool(
                os.environ.get("ZENITH_ALLOW_AGENT_TERMINAL"), default=False
            ),
            allow_unsandboxed=_resolve_bool(
                os.environ.get("ZENITH_ALLOW_UNSANDBOXED"), default=False
            ),
        )

    # ------------------------------------------------------------------
    # Provider accessors
    # ------------------------------------------------------------------

    @property
    def orchestrator_provider(self):
        return get_provider(self.orchestrator_provider_name)

    @property
    def worker_provider(self):
        return get_provider(self.worker_provider_name)

    @property
    def validator_provider(self):
        return get_provider(self.validator_provider_name or self.worker_provider_name)

    @property
    def terminal_reviewer_provider(self):
        name = (
            self.terminal_reviewer_provider_name
            or self.validator_provider_name
            or self.worker_provider_name
        )
        return get_provider(name)

    @property
    def resolved_worker_acp_command(self) -> str | None:
        return self.worker_acp_command or self.worker_provider.default_worker_acp_command

    @property
    def resolved_validator_acp_command(self) -> str | None:
        """Explicit override, else inherit; provider default only on a
        provider switch.

        Mirrors `ProviderSelection.resolved_validation_worker_acp_command`
        (providers.py): a custom worker command must cascade to a validator
        on the *same* provider. The provider default is a fallback for a
        *different* validator provider, not a shadow over the inherited
        command — `default_worker_acp_command` is always truthy, so putting
        it ahead of inheritance in a plain or-chain silently discards the
        user's custom command (model flags, wrapper scripts, mocks).
        """
        if self.validator_acp_command:
            return self.validator_acp_command
        if self.validator_provider.name != self.worker_provider.name:
            return (
                self.validator_provider.default_worker_acp_command
                or self.resolved_worker_acp_command
            )
        return (
            self.resolved_worker_acp_command
            or self.validator_provider.default_worker_acp_command
        )

    @property
    def resolved_terminal_reviewer_acp_command(self) -> str | None:
        """Same cascade as the validator, one level up: inherit the
        validator's resolved command unless the reviewer switches provider.
        """
        if self.terminal_reviewer_acp_command:
            return self.terminal_reviewer_acp_command
        if self.terminal_reviewer_provider.name != self.validator_provider.name:
            return (
                self.terminal_reviewer_provider.default_worker_acp_command
                or self.resolved_validator_acp_command
            )
        return (
            self.resolved_validator_acp_command
            or self.terminal_reviewer_provider.default_worker_acp_command
        )

    @property
    def provider_selection(self) -> ProviderSelection:
        return ProviderSelection(
            orchestrator=self.orchestrator_provider,
            worker=self.worker_provider,
            validation_worker=(
                self.validator_provider
                if self.validator_provider_name
                else None
            ),
            worker_acp_command=self.worker_acp_command,
            validation_worker_acp_command=self.validator_acp_command,
            terminal_reviewer=(
                self.terminal_reviewer_provider
                if self.terminal_reviewer_provider_name
                else None
            ),
            terminal_reviewer_acp_command=self.terminal_reviewer_acp_command,
        )

    # ------------------------------------------------------------------
    # Bucket paths
    # ------------------------------------------------------------------

    def bucket_root(self, project_id: str) -> Path:
        """Per-project root: parent of .zenith/ and .zenith-runtime/.

        `project_id` comes from an MCP tool argument (i.e. from an agent) and is
        joined into a path, so it is validated here — the single choke point for
        every bucket path in the store.
        """
        return self.projects_dir / safe_path_component(project_id, kind="project_id")

    def zenith_dir(self, project_id: str) -> Path:
        """Durable, all-roles-readable record (brief, decisions, skills, missions)."""
        return self.bucket_root(project_id) / ".zenith"

    def zenith_runtime_dir(self, project_id: str) -> Path:
        """Orchestrator-only cursors (project.json, state.json, dag.json, ...)."""
        return self.bucket_root(project_id) / ".zenith-runtime"

    def skill_dirs(self, project_id: str | None = None) -> list[Path]:
        dirs: list[Path] = []
        if project_id is not None:
            dirs.append(self.zenith_dir(project_id) / "skills")
        dirs.append(self.harness_home / "skills")
        dirs.append(self.bundled_dir / "skills")
        return dirs

    # ------------------------------------------------------------------
    # Role-specialized variants
    # ------------------------------------------------------------------

    def for_role(
        self, role: Literal["worker", "validator", "terminal_reviewer"]
    ) -> HarnessConfig:
        if role == "worker":
            return self
        if role == "validator":
            return replace(
                self,
                worker_provider_name=(
                    self.validator_provider_name or self.worker_provider_name
                ),
                worker_acp_command=self.resolved_validator_acp_command,
                worker_reasoning_effort=(
                    self.validator_reasoning_effort or self.worker_reasoning_effort
                ),
            )
        if role == "terminal_reviewer":
            return replace(
                self,
                worker_provider_name=(
                    self.terminal_reviewer_provider_name
                    or self.validator_provider_name
                    or self.worker_provider_name
                ),
                worker_acp_command=self.resolved_terminal_reviewer_acp_command,
                worker_reasoning_effort=(
                    self.terminal_reviewer_reasoning_effort
                    or self.validator_reasoning_effort
                    or self.worker_reasoning_effort
                ),
            )
        raise ValueError(f"unknown role: {role}")

    # ------------------------------------------------------------------
    # Safety accessors
    # ------------------------------------------------------------------

    def effective_runtime_mode(self) -> str | None:
        """The ACP runtime mode to request, honouring the sandbox opt-in.

        Providers declare `acp_runtime_mode` (claude: `bypassPermissions`).
        Requesting it is an unrestricted-execution decision, so it is withheld
        unless the operator opted in; the adapter then applies its own default
        permission mode instead of the harness silently disabling approvals.
        """
        mode = self.worker_provider.acp_runtime_mode
        if not mode:
            return None
        return mode if self.allow_unsandboxed else None

    def terminal_available(self) -> bool:
        """Whether ACP `terminal/*` (agent command execution) is granted."""
        return self.allow_agent_terminal

    def clamp_terminal_output(self, requested: object) -> int:
        """Clamp an agent-supplied `outputByteLimit` to the configured ceiling.

        The value arrives from the agent and previously became an unbounded
        in-memory buffer, so a worker could ask for gigabytes.
        """
        limit = self.max_terminal_output_bytes
        if isinstance(requested, bool) or not isinstance(requested, int):
            return min(1024 * 1024, limit)
        if requested <= 0:
            return min(1024 * 1024, limit)
        return min(requested, limit)
