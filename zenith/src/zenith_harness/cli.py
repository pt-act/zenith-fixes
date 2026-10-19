from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import click

from .assets import AssetLoader, iter_skill_directories
from .config import VALID_REASONING_EFFORTS, HarnessConfig
from .envelope import render_task_list
from .providers import (
    ProviderDefinition,
    ProviderSelection,
    default_worker_provider_name,
    get_provider,
    provider_names_for_role,
)
from .storage import ProjectStore

RUNTIME_ENV_FORWARD_ALLOWLIST = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    "CLAUDE_CODE_EFFORT_LEVEL",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "CLAUDE_CODE_SUBAGENT_MODEL",
    "GLM_API_KEY",
    "GLM_BASE_URL",
    "MAX_THINKING_TOKENS",
    "ZENITH_WORKER_REASONING_EFFORT",
    "ZENITH_VALIDATOR_REASONING_EFFORT",
    "ZENITH_TERMINAL_REVIEWER_REASONING_EFFORT",
    "ZAI_API_KEY",
    "ZAI_BASE_URL",
)

# Subset of the allowlist that is safe to persist to disk. Names ending in
# _API_KEY / _AUTH_TOKEN are secrets and are never written to the workspace;
# see `_forwarded_runtime_env`.
_SECRET_ENV_SUFFIXES = ("_API_KEY", "_AUTH_TOKEN", "_TOKEN", "_SECRET", "_PASSWORD")


def _echo_text(text: str, *, err: bool = False) -> None:
    """Print text, degrading gracefully on a non-UTF-8 terminal.

    The task-list render uses `—`, `→`, and `←`. On a `LC_ALL=C` terminal or a
    log captured in a legacy encoding those become mojibake, so fall back to
    ASCII equivalents when the stream cannot encode them.
    """
    stream = sys.stderr if err else sys.stdout
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = (
            text.replace("→", "->")
            .replace("←", "<-")
            .replace("—", "--")
            .replace("…", "...")
        )
    click.echo(text, err=err)


# Human-readable labels for the runtime's snake_case state tokens. The wire
# value stays machine-facing; the CLI is read by a person.
_STATE_LABELS = {
    "draft": "draft (no plan submitted)",
    "mission_planning": "planning (write the contract, then submit_plan)",
    "mission_running": "running",
    "attention_needed": "attention needed (decide every open item)",
    "done": "done",
    "failed": "failed",
    "aborted": "mission stopped",
}


def _state_label(state) -> str:
    """Render a project state for a human, keeping the raw token visible."""
    token = getattr(state, "state", None) or "draft"
    label = _STATE_LABELS.get(token)
    return f"{token} — {label}" if label else str(token)


@click.group()
def cli() -> None:
    """Zenith CLI — set up + inspect long-running coding projects."""


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--agent",
    type=click.Choice(provider_names_for_role("orchestrator")),
    default=None,
    help="Convenience: sets orchestrator+worker provider in one shot.",
)
@click.option(
    "--orchestrator-provider",
    type=click.Choice(provider_names_for_role("orchestrator")),
    default=None,
)
@click.option(
    "--worker-provider",
    type=click.Choice(provider_names_for_role("worker")),
    default=None,
)
@click.option("--worker-acp-command", default=None)
@click.option("--validator-provider", type=click.Choice(provider_names_for_role("worker")), default=None)
@click.option("--validator-acp-command", default=None)
@click.option("--terminal-reviewer-provider", type=click.Choice(provider_names_for_role("worker")), default=None)
@click.option("--terminal-reviewer-acp-command", default=None)
@click.option("--worker-reasoning-effort", type=click.Choice(VALID_REASONING_EFFORTS), default=None)
@click.option("--validator-reasoning-effort", type=click.Choice(VALID_REASONING_EFFORTS), default=None)
@click.option("--terminal-reviewer-reasoning-effort", type=click.Choice(VALID_REASONING_EFFORTS), default=None)
@click.option("--zenith-home", type=click.Path(), default=None)
@click.option("--workspace-dir", "workspace_dir", type=click.Path(exists=True), default=".")
def init(
    agent: str | None,
    orchestrator_provider: str | None,
    worker_provider: str | None,
    worker_acp_command: str | None,
    validator_provider: str | None,
    validator_acp_command: str | None,
    terminal_reviewer_provider: str | None,
    terminal_reviewer_acp_command: str | None,
    worker_reasoning_effort: str | None,
    validator_reasoning_effort: str | None,
    terminal_reviewer_reasoning_effort: str | None,
    zenith_home: str | None,
    workspace_dir: str,
) -> None:
    """Initialize host-agent surface: MCP/codex config + provider agents + orchestrator prompt.

    The project bucket is created by `start_project` at the first MCP call from
    the host agent — `zenith init` stages the host-agent-facing files the agent
    loads at startup (provider configs, subagent definitions, orchestrator
    prompt, and the bundled skill surface under the provider's skill dirs). The
    workspace stays clean of `.zenith/` until `start_project` runs.
    """
    workspace = Path(workspace_dir).resolve()
    config = HarnessConfig.discover()
    loader = AssetLoader(config)
    selection = _resolve_selection(
        agent=agent,
        orchestrator=orchestrator_provider,
        worker=worker_provider,
        worker_acp_command=worker_acp_command,
        validator=validator_provider,
        validator_acp_command=validator_acp_command,
        terminal_reviewer=terminal_reviewer_provider,
        terminal_reviewer_acp_command=terminal_reviewer_acp_command,
    )

    # 1) MCP / Codex config
    storage_env = _storage_env(zenith_home=zenith_home, workspace=workspace, selection=selection)
    # Flags are sugar for the ZENITH_*_REASONING_EFFORT env vars and win over
    # valid inherited shell settings. An invalid value already in the
    # environment still fails fast at discover() above — flags override
    # settings, they don't mask broken ones (the same validation would raise
    # at server launch anyway).
    effort_env = {
        var: value
        for var, value in (
            ("ZENITH_WORKER_REASONING_EFFORT", worker_reasoning_effort),
            ("ZENITH_VALIDATOR_REASONING_EFFORT", validator_reasoning_effort),
            ("ZENITH_TERMINAL_REVIEWER_REASONING_EFFORT", terminal_reviewer_reasoning_effort),
        )
        if value
    }
    _write_bootstrap_config(workspace, selection, storage_env, effort_env)
    _ensure_workspace_gitignore(workspace)

    # 2) Per-provider agents + orchestrator prompt
    for provider in selection.providers():
        _setup_provider_assets(workspace, loader, provider)

    click.echo(
        f"\nInitialized v5 project workspace at {workspace}: "
        f"orchestrator={selection.orchestrator.name}, "
        f"worker={selection.worker.name}, "
        f"validator={selection.resolved_validation_worker.name}."
    )
    click.echo(
        "Bucket lives at $ZENITH_HOME/projects/<pid>/ — created on the first "
        "`start_project(brief, workspace_dir)` call."
    )
    _echo_next_steps(selection.orchestrator)


# ---------------------------------------------------------------------------
# install-skills
# ---------------------------------------------------------------------------


@cli.command("install-skills")
@click.option("--target", type=click.Path(), required=True)
def install_skills_cmd(target: str) -> None:
    """Install bundled skills to a target directory (e.g. <ws>/.zenith/skills/)."""
    config = HarnessConfig.discover()
    loader = AssetLoader(config)
    _copy_skills(loader, Path(target).resolve())
    click.echo(f"Installed bundled skills to {target}")


# ---------------------------------------------------------------------------
# list-projects
# ---------------------------------------------------------------------------


@cli.command("list-projects")
def list_projects_cmd() -> None:
    """List all projects in HARNESS bucket."""
    store = ProjectStore(HarnessConfig.discover())
    projects = store.list_projects()
    if not projects:
        click.echo("No projects.")
        return
    for p in projects:
        click.echo(f"  {p.id}   ws={p.workspace_dir}   created={p.created_at}")


# ---------------------------------------------------------------------------
# show-project
# ---------------------------------------------------------------------------


@cli.command("show-project")
@click.argument("project_id")
def show_project_cmd(project_id: str) -> None:
    """Show envelope + compact task list for a project."""
    store = ProjectStore(HarnessConfig.discover())
    try:
        record = store.load_project(project_id)
    except FileNotFoundError:
        raise click.ClickException(f"Project not found: {project_id}")
    state = store.load_state(project_id)
    click.echo(f"id:        {record.id}")
    click.echo(f"workspace: {record.workspace_dir}")
    click.echo(f"created:   {record.created_at}")
    click.echo(f"state:     {_state_label(state)}")
    mid = record.current_mission_id
    if mid:
        click.echo(f"mission:   {mid}")
        try:
            tl = store.load_task_list(record.id, mid)
            ts = store.load_task_state(record.id, mid)
            rendered = render_task_list(tl, ts)
            if rendered:
                click.echo("")
                _echo_text(rendered)
        except FileNotFoundError:
            click.echo("  (task list not yet submitted)")


# ---------------------------------------------------------------------------
# inspect-tasks
# ---------------------------------------------------------------------------


@cli.command("inspect-tasks")
@click.option("--project", "project_id", required=True)
@click.option("--mission", "mission_id", default=None)
def inspect_tasks_cmd(project_id: str, mission_id: str | None) -> None:
    """Render the compact text task list for a mission."""
    store = ProjectStore(HarnessConfig.discover())
    if mission_id is None:
        mid_list = store.list_missions(project_id)
        if not mid_list:
            raise click.ClickException("no missions in this project")
        mission_id = mid_list[-1]
    try:
        tl = store.load_task_list(project_id, mission_id)
    except FileNotFoundError:
        raise click.ClickException(f"tasks.json not found for {project_id}/{mission_id}")
    ts = store.load_task_state(project_id, mission_id)
    rendered = render_task_list(tl, ts, mode="full")
    if rendered:
        _echo_text(rendered)


# ---------------------------------------------------------------------------
# abort-project
# ---------------------------------------------------------------------------


@cli.command("abort-project")
@click.argument("project_id")
@click.option("--reason", required=True)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip the confirmation prompt (for scripts).",
)
def abort_project_cmd(project_id: str, reason: str, yes: bool) -> None:
    """Mark a project Aborted (CLI-side: preserves tasks.json + attempts/)."""
    from .controller import build_controller

    if not yes:
        click.confirm(
            f"Abort {project_id}? This is terminal and cannot be undone.",
            abort=True,
        )
    config = HarnessConfig.discover()
    # Same construction as the MCP server (M-9); abort never dispatches, but
    # sharing the factory keeps the two surfaces from drifting.
    controller = build_controller(config)
    env = controller.abort_project(project_id, reason)
    click.echo(f"Aborted {project_id}: state={_state_label(env.state)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _copy_skills(loader: AssetLoader, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    bundled = loader.bundled_skills_dir()
    if not bundled.exists():
        click.echo(f"warning: bundled skills not found at {bundled}", err=True)
        return
    for skill_dir in iter_skill_directories(bundled):
        dest = target / skill_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(skill_dir / "SKILL.md", dest / "SKILL.md")


def _echo_next_steps(orchestrator: ProviderDefinition) -> None:
    prompt_path = orchestrator.orchestrator_prompt_output_path
    click.echo("")
    click.echo("Next:")
    click.echo("  1. Start your agent from the initialized project workspace:")
    click.echo(f"     {orchestrator.name}")
    if prompt_path:
        click.echo("  2. Ask it:")
        click.echo(
            f"     First read {prompt_path} and treat it as your primary role, then use Zenith to run this mission."
        )
        click.echo("")
        click.echo("     <your instruction or query>")


def _resolve_selection(
    *,
    agent: str | None,
    orchestrator: str | None,
    worker: str | None,
    worker_acp_command: str | None,
    validator: str | None,
    validator_acp_command: str | None,
    terminal_reviewer: str | None,
    terminal_reviewer_acp_command: str | None,
) -> ProviderSelection:
    if agent and orchestrator and agent != orchestrator:
        raise click.UsageError("--agent conflicts with --orchestrator-provider")
    orch = orchestrator or agent or "claude"
    wrk = worker or (agent if agent in provider_names_for_role("worker") else None) or default_worker_provider_name(orch)
    return ProviderSelection(
        orchestrator=get_provider(orch),
        worker=get_provider(wrk),
        validation_worker=get_provider(validator) if validator else None,
        worker_acp_command=worker_acp_command,
        validation_worker_acp_command=validator_acp_command,
        terminal_reviewer=get_provider(terminal_reviewer) if terminal_reviewer else None,
        terminal_reviewer_acp_command=terminal_reviewer_acp_command,
    )


def _storage_env(
    *,
    zenith_home: str | None,
    workspace: Path,
    selection: ProviderSelection,
) -> dict[str, str]:
    env: dict[str, str] = {}
    if zenith_home:
        env["ZENITH_HOME"] = str(Path(zenith_home).expanduser().resolve())
    return env


def _forwarded_runtime_env() -> dict[str, str]:
    """Environment passed to the MCP server it spawns.

    Only *non-secret* configuration is forwarded. The provider credentials in
    `RUNTIME_ENV_FORWARD_ALLOWLIST` are deliberately **not** copied here: they
    used to be written verbatim into `<workspace>/.mcp.json`, i.e. into the
    user's own repository, where the next `git add -A` committed them. The MCP
    server inherits the orchestrator's environment at spawn time instead, so it
    still sees the keys it needs without them ever touching disk.
    """
    return {
        key: value
        for key in RUNTIME_ENV_FORWARD_ALLOWLIST
        if not key.endswith(_SECRET_ENV_SUFFIXES) and (value := os.environ.get(key))
    }


# Paths `zenith init` creates inside the user's workspace. They are
# workspace-local agent plumbing, not project content, and `.mcp.json` in
# particular can carry environment configuration, so they are gitignored.
_WORKSPACE_LOCAL_PATHS = (
    ".mcp.json",
    ".agents/",
    ".claude/settings.json",
    ".claude/agents/",
    ".codex/config.toml",
    ".codex/agents/",
    ".zenith-review-*/",
)


def _restrict_permissions(path: Path) -> None:
    """Best-effort 0600 on a config file that may carry credentials."""
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _ensure_workspace_gitignore(workspace: Path) -> None:
    """Append Zenith's workspace-local paths to `.gitignore`.

    `zenith init` writes agent configuration into the repository the user
    pointed at. Leaving those files untracked-but-visible meant a routine
    `git add -A` staged them — historically including live API keys.
    """
    gitignore = workspace / ".gitignore"
    try:
        existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    except OSError:
        return
    missing = [
        entry
        for entry in _WORKSPACE_LOCAL_PATHS
        if entry not in {line.strip() for line in existing.splitlines()}
    ]
    if not missing:
        return
    block = "\n# Zenith workspace-local agent plumbing\n" + "\n".join(missing) + "\n"
    try:
        if existing and not existing.endswith("\n"):
            existing += "\n"
        gitignore.write_text(existing + block, encoding="utf-8")
        click.echo(f"Updated {gitignore}")
    except OSError as exc:
        click.echo(f"warning: could not update {gitignore}: {exc}", err=True)


def _zenith_project_root() -> Path:
    """Return the source checkout that owns the Zenith runtime uv project."""
    start = Path(__file__).resolve()
    for candidate in (start.parent, *start.parents):
        pyproject = candidate / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            continue
        if 'name = "zenith-harness"' in text:
            return candidate
    raise click.ClickException(
        f"Could not locate the Zenith uv project root from {start}: no parent "
        "directory has a pyproject.toml declaring name = \"zenith-harness\". "
        "Run `zenith init` from the source checkout (or a copy of it); the "
        "generated .mcp.json needs that path to launch zenith-server."
    )


def _resolve_acp_command(command: str) -> str:
    """Resolve an ACP adapter command to something spawnable.

    A bare name is resolved through PATH so the spawned worker does not depend
    on the orchestrator's PATH surviving into a desktop-launched session, and a
    missing adapter produces a clear error instead of a node that silently
    synthesizes a "missing handoff".
    """
    if not command:
        return command
    try:
        if os.name == "nt":
            from .acp_runner import _split_command

            parts = _split_command(command, posix=False)
        else:
            parts = shlex.split(command)
    except ValueError:
        # An unbalanced quote is an operator error; leave the string untouched
        # and let the spawn path report it rather than silently rewriting it.
        return command
    if not parts:
        return command
    executable = shutil.which(parts[0])
    if executable is None:
        return command
    # Re-quote arguments so the later exec-side split reconstructs the same
    # argv (paths with spaces survive). Windows needs list2cmdline, not POSIX
    # single quotes.
    argv = [executable, *parts[1:]]
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _mcp_server_args() -> list[str]:
    return [
        "run",
        "--project",
        str(_zenith_project_root()),
        "zenith-server",
        "--mode",
        "orchestrator",
    ]


def _write_bootstrap_config(
    workspace: Path,
    selection: ProviderSelection,
    storage_env: dict[str, str],
    cli_env: dict[str, str],
) -> None:
    fmt = selection.orchestrator.config_format
    env = {**selection.env(), **storage_env, **_forwarded_runtime_env(), **cli_env}
    # Resolve bare adapter names against PATH once, at init time.
    for key in (
        "ZENITH_WORKER_ACP_COMMAND",
        "ZENITH_VALIDATOR_ACP_COMMAND",
        "ZENITH_TERMINAL_REVIEWER_ACP_COMMAND",
    ):
        if key in env:
            env[key] = _resolve_acp_command(env[key])
    server_args = _mcp_server_args()
    if fmt == "mcp_json":
        path = workspace / ".mcp.json"
        existing = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )
        existing.setdefault("mcpServers", {})["zenith"] = {
            "type": "stdio",
            "command": "uv",
            "args": server_args,
            "env": env,
        }
        path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        _restrict_permissions(path)
        click.echo(f"Wrote {path}")
    elif fmt == "codex_config":
        config_path = workspace / ".codex" / "config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        # TOML basic strings share JSON's escape syntax, so json.dumps closes
        # and escapes embedded quotes. ACP commands splice config via
        # `-c key="value"`, and forwarded env values may hold quotes or
        # backslashes; interpolating either raw emits invalid TOML.
        env_lines = "\n".join(f"{k} = {json.dumps(v)}" for k, v in env.items())
        block = (
            '[features]\n'
            'memories = true\n'
            "# BEGIN zenith\n"
            "[mcp_servers.zenith]\n"
            'command = "uv"\n'
            f"args = {json.dumps(server_args)}\n"
            "startup_timeout_sec = 10\n"
            "tool_timeout_sec = 1000000\n"
            "\n"
            "[mcp_servers.zenith.env]\n"
            f"{env_lines}\n"
            "# END zenith\n"
        )
        _replace_managed_block(config_path, "# BEGIN zenith", "# END zenith", block)
        click.echo(f"Wrote {config_path}")
    else:
        raise ValueError(f"unsupported config_format: {fmt}")


def _replace_managed_block(path: Path, start: str, end: str, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in existing and end in existing:
        before, _, tail = existing.partition(start)
        _, _, after = tail.partition(end)
        updated = before.rstrip()
        if updated:
            updated += "\n\n"
        updated += block.rstrip() + "\n"
        if after.strip():
            updated += "\n" + after.lstrip("\n")
    else:
        updated = existing.rstrip()
        if updated:
            updated += "\n\n"
        updated += block.rstrip() + "\n"
    path.write_text(updated, encoding="utf-8")


def _setup_provider_assets(
    workspace: Path,
    loader: AssetLoader,
    provider: ProviderDefinition,
) -> None:
    if provider.agent_output_dir:
        agents_dir = workspace / provider.agent_output_dir
        _copy_provider_agents(
            loader, agents_dir, provider.name
        )
        click.echo(f"Installed {provider.name} subagents to {agents_dir}")
    # Install bundled skills into the host-agent skill surface so the
    # orchestrator can discover playbooks/skills at startup — `start_project`
    # runs only after the host agent is already up, so the surface must exist
    # before the first MCP call. `start_project` later merges bucket skills
    # (including project-authored ones) into these dirs.
    for rel in provider.skill_dirs:
        dest = workspace / rel
        _copy_skills(loader, dest)
        click.echo(f"Installed bundled skills to {dest}")
    if provider.orchestrator_prompt_output_path:
        path = workspace / provider.orchestrator_prompt_output_path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            body = loader.load_prompt_file("orchestrator", "system_prompt.md")
            path.write_text(body, encoding="utf-8")
            click.echo(f"Created {path}")


def _copy_provider_agents(loader: AssetLoader, target: Path, provider_name: str) -> None:
    bundled = loader.bundled_agents_dir(provider_name)
    if not bundled.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for agent_file in sorted(bundled.glob("*")):
        if agent_file.is_file():
            shutil.copy2(agent_file, target / agent_file.name)
