# Security Policy

## Reporting a vulnerability

Please do **not** report security vulnerabilities through public GitHub issues.

Instead, use [GitHub's private vulnerability reporting](https://github.com/Intelligent-Internet/zenith/security/advisories/new) for this repository. We will acknowledge your report within a few business days and keep you informed as we investigate and remediate.

## Scope

Zenith runs coding agents that can execute commands and modify files in the workspace you point them at. When reporting, issues of particular interest include:

- Escapes from the configured workspace directory by workers, testers, or validators
- Prompt-injection paths that cause the orchestrator to take actions outside its mission
- Unsafe handling of ACP subprocess input/output
- Forged node verdicts (worker / terminal-reviewer handoffs)

## Enforcement model

Zenith is a harness for **untrusted** agents: a worker is treated as an
adversary that will attempt to escape its sandbox, read credentials, and report
success it did not achieve. The following boundaries are enforced in code, and
each has a regression test in `tests/test_audit_regressions.py` or
`tests/test_audit_gaps.py`.

### Workspace confinement

ACP `fs/read_text_file` and `fs/write_text_file` requests are resolved and then
required to stay inside the session workspace. `../` traversal, absolute paths,
and symlinked parents that escape are all refused.

### Workspace mutation is opt-in

`start_project` / `advance_project` do **not** write into the user's repository.
The host-agent skill shims (`.agents/skills`, `.claude/skills`,
`.codex/skills`, `AGENTS.md`) are created only when the operator asks for them:

```bash
export ZENITH_WORKSPACE_SHIMS=1
```

Replacing a real (bootstrap-only) skill directory with a symlink deletes it, so
that additionally requires `ZENITH_REPLACE_BOOTSTRAP_SKILLS=1`; without it the
two trees are merged and the directory is preserved. `zenith init` still writes
the configuration files the operator explicitly ran it to create.

### ACP agent environment is allowlisted

The worker/validator/reviewer ACP process is spawned with a credential-free
base environment (`PATH`, `HOME`, `LANG`, `TERM`, `TMPDIR`, proxy/CA settings)
plus only the selected provider's own namespace: `OPENAI_*`/`CODEX_*` for
codex, `ANTHROPIC_*`/`CLAUDE_*` for claude, `HERMES_*` (plus common provider
keys) for hermes. Unrelated ambient credentials (AWS, GitHub, Slack, ...) are
not inherited. An operator who needs one more variable can name it in
`ZENITH_ACP_ENV_ALLOW` (comma-separated). The per-node MCP server subprocess
gets only the base environment plus its `ZENITH_*` configuration.

### Concurrent orchestrators are serialized

Every mutating tool call takes an OS-level per-project lock
(`.zenith-runtime/project.lock`, `flock`/`msvcrt`). A second orchestrator
process on the same project waits up to 60 seconds and then receives
`project_busy`; it cannot interleave a duplicate dispatch or a lost update.
`inspect_project` takes the same lock, so it never returns a torn snapshot.

### Decision commits are journaled

`decide_attention` writes a pre-image journal before its first mutation. If a
write fails (disk full, permission error) or the process dies mid-commit, the
next tool call restores the snapshot and removes the journal, so the same
decision can be retried instead of failing permanently on `task_id_collision`.

### Closure is fail-closed

The terminal reviewer can veto a close but cannot grant one by itself. Runtime
also checks that the state is `mission_running`, no attention is open, every
task/gate is cleared or superseded, every contract assertion passed, and the
reviewer's cited evidence was actually read through the ACP channel. Mission
closure also requires `closure.json` with at least one acceptance command; a
missing or empty file blocks closure. Its commands run without a shell, with a
credential-free environment and a timeout; any non-zero exit blocks closure. A
prompt-injected reviewer that says `done=true` therefore cannot seal a mission
whose machine checks fail.

### OS-level sandbox (opt-in)

An ACP agent can be confined with an unprivileged Linux sandbox. `auto`
prefers `bwrap`, then the built-in `unshare` + `pivot_root` backend, and fails
closed if neither is usable:

```bash
export ZENITH_SANDBOX=auto       # or bwrap / unshare / none
```

The sandbox binds the mission workspace read-write at its absolute path,
system trees read-only, `/proc`/`/tmp`/`/dev` fresh, and a per-node home
containing only the selected provider's config. The host home, `$ZENITH_HOME`,
other projects, and unrelated credentials are absent. ACP `terminal/*`
commands run inside the same namespace.

Phase 1 keeps host networking because the per-node MCP server is on host
loopback; `ZENITH_SANDBOX_NETWORK=none` is refused until a container/proxy
backend exists. `ZENITH_SANDBOX=none` is an explicit opt-out with no boundary
beyond the environment allowlist and workspace confinement.

### ACP launch is exec, never a shell

The adapter command is configuration, but it is not code: it is split with
`shlex` and passed to `create_subprocess_exec`. Shell metacharacters in the
command (or in any value spliced into it) are literal argv tokens the adapter
rejects, not commands the host runs.

### Agent command execution is opt-in

ACP `terminal/*` lets an agent run arbitrary commands. That capability cannot be
confined to the workspace by this client (arbitrary `cwd`, inherited
environment), so it is **refused by default** and advertised as unavailable in
the `initialize` capabilities. Enable it only for agents you trust:

```bash
export ZENITH_ALLOW_AGENT_TERMINAL=1
```

When enabled, commands run via `create_subprocess_exec` (never a shell, so
metacharacters are not interpreted), `cwd` is confined to the workspace, and the
child environment is built from a small allowlist (`PATH`, `HOME`, `LANG`,
`TERM`, ...) — provider credentials in the orchestrator's environment are not
passed through.

### Permission requests are denied by default

Every ACP `session/request_permission` is answered with a denial unless a policy
says otherwise. Unattended runs that need automatic approval must opt in:

```bash
export ZENITH_ACP_PERMISSION_POLICY=allow_all   # or allow_read
```

### Unrestricted agent execution is opt-in

By default Zenith does **not** disable the agent's own sandbox or approval gate,
and it does not request an unrestricted ACP runtime mode (`bypassPermissions`).
Codex keeps its sandbox and `approval_policy`; Claude Code keeps its default
permission mode. `zenith init` no longer writes `sandbox_mode` into the
generated `.codex/config.toml` — if an earlier init left
`sandbox_mode = "danger-full-access"` there, delete that line to restore
Codex's own default. To run fully unattended — accepting that an agent can
execute anything the user can — set:

```bash
export ZENITH_ALLOW_UNSANDBOXED=1
```

Only set this inside a disposable container or VM if the workspace contains
anything you would not hand to an arbitrary program.

### Node verdicts are authenticated

Each worker / terminal-reviewer MCP server is spawned with a per-node random
bearer token and refuses unauthenticated requests (HTTP 401). Without this, any
local process could POST a forged `end_node` / `submit_terminal_review` to the
node's ephemeral port and write a passing verdict the coordinator trusts. The
token is generated per node, passed via `ZENITH_MCP_AUTH_TOKEN`, and never
persisted to disk.

A validation verdict is also restricted to the assertions the task was assigned
(`ZENITH_NODE_TARGETS`): a validator cannot report coverage for a contract
target it was never asked to verify.

### Credentials

`zenith init` never writes provider API keys into the workspace. Only non-secret
configuration (base URLs, model names, reasoning effort) is persisted; the MCP
server inherits secrets from the orchestrator's environment at spawn time.
Generated agent configuration (`.mcp.json`, `.codex/config.toml`,
`.claude/settings.json`, provider agent/skill directories) is added to the
workspace `.gitignore` and written `0600`, so a routine `git add -A` cannot
stage it.

If you ran an earlier version of `zenith init` with credentials in the
environment, **rotate those keys**: the values were written to `.mcp.json` in
cleartext.

## Dependency and release policy

Runtime dependencies are pinned to compatible-minor ranges (`~=`) and resolved
through the committed `uv.lock`; the documented install path and CI both run
`uv sync --frozen`, so a build uses the reviewed lockfile rather than resolving
new versions. Dependency bumps are reviewed changes, not automatic updates.
Releases are tagged from `main` after the full test suite passes; security fixes
land on the latest tag.

## Supported versions

Security fixes are applied to the latest release on `main`.
