# ADR 001: Live Repo Context & AST Indexing vs Naive File Dumps

## Status
Accepted

## Context
When building an autonomous coding agent, LLMs require codebase context to understand project topology, module boundaries, and symbol references. Naive approaches inject entire source files or raw directory tree dumps into the system prompt. For repositories with more than a few files, this consumes enormous token budgets, triggers attention degradation ("lost in the middle"), increases time-to-first-token (TTFT), and quickly blows past context windows.

## Decision
We decouple repo context into two complementary layers:
1. **Lightweight Workspace Snapshot:** Formatted markdown injected upfront containing current git branch, `git status --short`, repo instruction guidelines (`AGENTS.md`), and a bounded ASCII tree (depth ≤ 4, max 200 items, ignoring artifacts like `.git`, `node_modules`, `__pycache__`).
2. **Tree-sitter AST Symbol Outline with Vector Index (Qdrant):** Python source files are parsed into an Abstract Syntax Tree (AST) using Tree-sitter to extract discrete functional entities (`function_definition`, `class_definition`, `import_statement`) alongside exact span numbers (`start_line`, `end_line`). Symbols and code snippets are embedded and stored in Qdrant. The agent retrieves relevant symbols on demand via `search_symbols`.

## Consequences
- **Positive:** System prompt overhead is reduced by >85% compared to naive file injection. The agent gets pinpoint line locations of symbols without reading whole files.
- **Trade-off:** Requires a Tree-sitter grammar and local vector store (Qdrant).
