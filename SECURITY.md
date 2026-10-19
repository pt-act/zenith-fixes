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
each has a regression test in `tests/test_audit_regressions.py`.

### Workspace confinement

ACP `fs/read_text_file` and `fs/write_text_file` requests are resolved and then
required to stay inside the session workspace. `../` traversal, absolute paths,
and symlinked parents that escape are all refused.

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
permission mode. To run fully unattended — accepting that an agent can execute
anything the user can — set:

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

## Supported versions

Security fixes are applied to the latest release on `main`.
