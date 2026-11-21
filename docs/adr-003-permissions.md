# ADR 003: Structured Permissions, Path Jailing, and Ephemeral Docker Sandbox

## Status
Accepted

## Context
LLM coding agents inherently need to modify code and execute shell commands (e.g. running test suites, formatters, and git commands). Allowing unrestricted shell access or raw file writes creates catastrophic security risks:
- Prompt injections or model hallucinations can execute `rm -rf /` or leak environment secrets.
- Relative path escapes (`../../etc/passwd`) can compromise the host machine.
- Accidental background processes or network requests can corrupt external infrastructure.

## Decision
We enforce a strict three-tier defense-in-depth model:
1. **Path-Jailing File Tools:** Every filesystem operation (`read_file_lines`, `grep_search`, `apply_search_replace`) strictly resolves target paths against `workspace_root` and rejects any path escaping the root jail.
2. **Deterministic Search/Replace:** `apply_search_replace` requires an exact match that appears *exactly once* in the file. Matches of 0 or >1 occurrences are strictly rejected.
3. **Isolated Ephemeral Docker Sandbox:** Commands run inside an unprivileged Docker container with non-root user (`sandboxuser`, UID 1000), hard memory caps (`512m`), network completely disabled (`--network none`), and ephemeral lifecycles (`--rm`).
4. **LangGraph Human-in-the-Loop (`interrupt`):** In `"ask"` approval mode, any mutating or shell action pauses the LangGraph execution using `interrupt()`, awaiting explicit user confirmation or denial before execution proceeds.

## Consequences
- **Positive:** Zero risk of host system compromise or unintended external network access. Edits are deterministic and conflict-free.
- **Trade-off:** Running commands inside Docker containers adds ~100-300ms overhead per command compared to naked subprocess execution, which is an acceptable cost for complete process isolation.
