# ADR 005: Dual-Layer Memory Architecture & Durable Postgres Checkpoints

## Status
Accepted

## Context
Stateless agents lose all working context if the process crashes, the network disconnects, or the user pauses execution for approval. Furthermore, relying entirely on raw chat message history obscures the agent's current high-level goal, plan status, and modified files.

## Decision
We implement a dual-layer memory architecture:
1. **Durable Event Log Layer (`PostgresSaver`):** Checkpoints are committed to PostgreSQL after every graph node transition using `langgraph-checkpoint-postgres` with pooled connections. The entire state (messages, working memory, pending approvals, step counts) survives process restarts. Sessions are resumed simply by specifying the `thread_id`.
2. **Typed Working Memory Layer (`WorkingMemory`):** A typed Pydantic model (`current_goal`, `plan_steps`, `files_modified`, `read_file_hashes`, `last_verification_error`, `notes`) is injected fresh into the prompt each turn. Even if conversational turns are compacted or summarized, the agent retains exact structural knowledge of which files were modified and what verification errors occurred.

## Consequences
- **Positive:** Sessions are crash-resilient and auditable. State can be inspected via database queries or resumed anytime.
- **Trade-off:** Requires a PostgreSQL instance (provided via Docker container `postgres_db` on port 5432). A fallback to in-memory persistence is provided when database connectivity is unavailable.
