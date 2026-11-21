# ADR 006: Delegation via Bounded Read-Only Explorer Subagents

## Status
Accepted

## Context
When an agent needs to locate relevant code across a multi-directory codebase, executing multiple raw file reads and grep searches in the main agent loop severely bloats the primary conversation history. Worse, unrestricted subagent architectures risk infinite delegation loops and unconstrained execution costs.

## Decision
We introduce a bounded Explorer subagent (`subagents/explorer.py`) accessible via `delegate_to_explorer`:
1. **Isolated Read-Only Tools:** The Explorer subgraph is provisioned ONLY with `read_file_lines`, `grep_search`, and `search_symbols`. It has zero access to mutating tools (`apply_search_replace`, `run_command`) and cannot spawn subagents.
2. **Hard Recursion and Step Limits:** The Explorer enforces a strict step limit (`max_steps=5`, `recursion_limit=10`).
3. **Distilled Summary Interface:** Rather than returning raw tool messages to the parent graph, the Explorer synthesizes its findings into a concise, distilled technical brief (under 300-400 words) containing only relevant file paths, line ranges, and architectural patterns.

## Consequences
- **Positive:** Protects the main agent's token budget during exploratory phases. Completely eliminates multi-agent recursion hazards.
- **Trade-off:** Adds an extra subgraph invocation step before editing begins, which is mitigated by fast execution with targeted queries.
