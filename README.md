# Autonomous Coding Agent Platform

**Headless autonomous coding agent built around Sebastian Raschka's 6 harness components.**
**Stack:** Python + LangChain + LangGraph + FastAPI + PostgreSQL + Qdrant + Docker + Next.js.

This project is designed as a resume-grade build for AI Engineer and Forward Deployed Engineer roles. It is not a chatbot wrapper; it is a production-grade harness that turns an LLM into a reliable software engineering agent with safe execution, AST-level context retrieval, human-in-the-loop approvals, and deterministic evaluation.

---

## Target Architecture: Raschka's 6 Components → Production Stack

| # | Raschka Component | Minimal Concept | Production Implementation | What You Prove |
|---|-------------------|-----------------|---------------------------|----------------|
| 1 | Live Repo Context | Directory tree, git status, repo instructions collected upfront | Workspace snapshot collector + Tree-sitter AST symbol outline + Qdrant hybrid retrieval | You understand why naive file-dumping blows token budgets |
| 2 | Prompt Shape & Cache Reuse | Static prefix separated from changing turn state | Immutable SystemMessage prefix (rules + tool schemas + repo snapshot) + mutable messages reducer | You can optimize TTFT and inference cost via prompt-caching boundaries |
| 3 | Structured Tools & Permissions | Named JSON tools with path validation and ask/auto/never gates | Path-jailed file tools + Docker sandbox for shell + LangGraph `interrupt()` approval gates | You treat LLM code execution as a security boundary, not an afterthought |
| 4 | Minimizing Context Bloat | Clips long outputs, deduplicates reads, compresses older turns | Output head/tail clipping, read-cache dedup via file hashes, rolling summarization at ~75% token budget | You manage token budgets on long-horizon coding tasks |
| 5 | Structured Session Memory | Full transcript on disk + compact working memory | AsyncPostgresSaver durable event log + typed WorkingMemory Pydantic state (goal, plan, files_touched, errors) | You build durable, resumable agent state with auditability |
| 6 | Delegation with Bounded Subagents | Scoped helper subagents, read-only tools, hard max_steps limits | Read-only Explorer subgraph (`recursion_limit=10`, no write tools, no subagent-spawning) returning distilled summaries | You orchestrate multi-agent systems without infinite recursion |

---

## Repository Structure (Conceptual)

```
autonomous-coding-agent/
├── README.md
├── pyproject.toml              # uv/poetry dependencies: langgraph, langchain, fastapi, qdrant-client, tree-sitter, docker
├── docker-compose.yml          # PostgreSQL (checkpoints), Qdrant (code index), sandbox runner
├── docker/
│   └── Dockerfile.sandbox      # Isolated container with ruff, pytest, git; network disabled by default
├── src/
│   └── coding_agent/
│       ├── config.py           # Pydantic settings: model provider, token budgets, approval mode
│       ├── state.py            # AgentState TypedDict + WorkingMemory Pydantic model (Component 5)
│       ├── context/
│       │   ├── snapshot.py     # [1] git status, AGENTS.md, directory tree
│       │   └── indexer.py      # [1] Tree-sitter AST chunker + Qdrant hybrid search
│       ├── prompts/
│       │   └── builder.py      # [2] Cache-optimized static prefix + dynamic state assembler
│       ├── tools/
│       │   ├── filesystem.py   # [3] Path-jailed read_file_lines, grep_search, apply_search_replace
│       │   ├── sandbox.py      # [3] Docker container command runner for pytest, ruff, git diff
│       │   └── approvals.py    # [3] LangGraph interrupt() gate for risky writes/commands
│       ├── memory/
│       │   ├── compaction.py   # [4] Output clipping, read-cache dedup, rolling summarizer
│       │   └── checkpointer.py # [5] PostgresSaver setup + WorkingMemory updater
│       ├── subagents/
│       │   └── explorer.py     # [6] Bounded read-only Explorer subgraph
│       └── graph.py            # Main StateGraph wiring all nodes and conditional edges
├── api/
│   └── main.py                 # FastAPI + SSE streaming + HITL resume endpoint
└── evals/
    └── run_benchmark.py        # Automated pass@1 + token/latency benchmark runner over 15–20 GitHub issues
```

---

## Build Methodology: Maximum Learning

Follow this loop for every component. Do not treat this as a tutorial where you copy starter code and tweak it.

1. **Read the reference.** Study Raschka's `mini-coding-agent` source for that component and the LangGraph docs for the primitive you will use (`StateGraph`, `interrupt`, `checkpointer`, subgraph invocation).
2. **Write the test first.** Define what success looks like (e.g., `apply_search_replace` must refuse a path outside the workspace root; Explorer subgraph must stop after 5 steps).
3. **Build it standalone.** Make the component work in isolation before wiring it into the graph.
4. **Instrument it.** Count tokens, measure latency, log every tool call. You cannot optimize what you do not measure.
5. **Break it.** Feed it a malicious path, a 50,000-line file, a non-existent git repo. Observe how it fails.
6. **Integrate.** Wire it into the LangGraph state machine. Verify the previous component still works.
7. **Document.** Write a one-paragraph Architecture Decision Record (ADR) for why you built it this way. This becomes your interview story.

---

## Phase-by-Phase Roadmap

### Phase 0 — Prerequisites & Sandbox (Days 1–3)

**Goal:** A reproducible environment where you can run code safely and track costs.

Learning objectives:
- Python dependency management with `uv` or `poetry`
- Docker basics: build an image, run a container, mount volumes, set CPU/memory limits, disable external network
- PostgreSQL and Qdrant running via `docker-compose`
- LangGraph + LangChain + FastAPI installed; a hello-world graph that passes state between two nodes

Mini-builds:
1. A Dockerfile that installs Python, `ruff`, `pytest`, `git`, and runs as a non-root user.
2. A `docker-compose.yml` that starts Postgres (port 5432) and Qdrant (port 6333).
3. A one-file LangGraph script with two nodes (`greet`, `respond`) and a SQLite checkpointer (temporary, before Postgres) so you understand state passing.

**Deliverable:** `docker-compose.yml` + README explaining how to spin up the infrastructure.

**Success criteria:** You can `docker compose up` and run a LangGraph node that writes to Postgres.

---

### Phase 1 — Components 1, 2, 3: The Core ACI Loop (Week 1)

**Goal:** A terminal-based agent that reads files, greps code, applies search/replace edits, and runs shell commands inside a Docker sandbox, all behind approval gates.

Build order (strict sequence):

#### Step 1.1 — Component 1: Live Repo Context
Research first: Raschka's snapshot logic, `tree-sitter` Python bindings, Qdrant collection schema for code (dense vector + sparse/bm25 + payload metadata).

Build:
- `context/snapshot.py`: Collect git branch, `git status --short`, `AGENTS.md`/`CLAUDE.md` contents, bounded directory tree (ignore `.git`, `node_modules`, `__pycache__`, etc.).
- `context/indexer.py`: Parse `.py` files with Tree-sitter into function/class/import chunks. Index into Qdrant with payload (`file_path`, `symbol_name`, `symbol_type`, `start_line`, `end_line`).
- A script that prints the snapshot to stdout. Do not connect an LLM yet.

**Learning check:** Can you explain why a symbol graph is more useful than character-chunking for a 100-file repository?

#### Step 1.2 — Component 2: Cache-Optimized Prompt Builder
Research first: LangChain `SystemMessage`, `HumanMessage`, `AnyMessage`; LangGraph `add_messages` reducer; provider prompt-caching behavior.

Build:
- `prompts/builder.py`: `assemble_prompt_messages(state)` returning `[static_system_prefix, working_memory_block, *dynamic_messages]`.
- The static prefix never changes mid-session; the working memory block is injected fresh each turn.

**Learning check:** Can you draw where the cache boundary sits in the message array?

#### Step 1.3 — Component 3: Deterministic Tools & Permissions
Research first: `langchain_core.tools @tool` decorator, Pydantic path validation, `langgraph.types.interrupt`, Docker SDK container execution.

Build:
- `tools/filesystem.py`: `read_file_lines`, `grep_search`, `apply_search_replace`. Every path resolved against `workspace_root`; reject traversal (`..`, absolute paths outside root).
- `tools/sandbox.py`: Run a command string inside an ephemeral Docker container with network disabled, read-only workspace mount except a writable `/workspace` volume, `mem_limit="512m"`.
- `tools/approvals.py`: `approval_mode` ("ask" | "auto" | "never"). In "ask", call `interrupt()` before any write or shell command.

**Learning check:** Can you demonstrate that `apply_search_replace` refuses a search block that matches 0 or 2+ locations?

#### Step 1.4 — Wire the Graph
Build `graph.py`:
- `StateGraph` with `AgentState` (messages, workspace_root, repo_snapshot, approval_mode, working_memory, step_count, max_steps).
- Nodes: `bootstrap_context` (fills repo_snapshot), `agent_model` (calls LLM with tools), `ToolNode` (executes tools), `verify` (runs ruff/pytest in sandbox).
- Conditional edge: if a tool call requests write/shell and `approval_mode == "ask"`, route to an interrupt node; otherwise continue.

**Deliverable:** A CLI loop (`python -m coding_agent.cli`) where you type a task, the agent proposes edits, you approve or reject, and it runs tests inside Docker.

**Success criteria:** Agent can fix a typo in a local Python file, run `pytest` in Docker, and report pass/fail—all without leaving the workspace jail.

---

### Phase 2 — Components 4, 5, 6: Context Hygiene, Memory & Subagents (Week 2)

**Goal:** The agent survives long sessions, remembers what it changed, and delegates big searches without bloating the main context.

#### Step 2.1 — Component 4: Context Hygiene
Research first: Token counting with `tiktoken`, head/tail truncation, rolling summarization prompts.

Build:
- `memory/compaction.py`:
  - `clip_output(text, max_lines)`: keep head + tail, drop middle.
  - `dedupe_read(state, path)`: if file hash unchanged since last read, return a pointer instead of re-reading.
  - `should_compact(messages, token_budget_pct=0.75)`: trigger summarization.
  - `summarize_turn(messages, goal)`: LLM summarizes older turns preserving goal, modified files, latest traceback.

**Learning check:** Measure token count before and after compaction on a 20-turn conversation.

#### Step 2.2 — Component 5: Dual-Layer Memory
Research first: LangGraph `AsyncPostgresSaver`, `PostgresSaver` setup, Pydantic `BaseModel` validation.

Build:
- `memory/checkpointer.py`: Configure `AsyncPostgresSaver` with your Postgres connection string. Verify checkpoints are written after every node.
- `state.py`: Add `WorkingMemory` Pydantic model (`current_goal`, `plan_steps`, `files_modified`, `read_file_hashes`, `last_verification_error`).
- A session-resume script: kill the agent mid-task, restart, load checkpoint from Postgres, continue where it left off.

**Learning check:** Can you resume a session after killing the process and prove `files_modified` persisted exactly?

#### Step 2.3 — Component 6: Bounded Explorer Subagent
Research first: LangGraph subgraph invocation (`subgraph.invoke` / `subgraph.stream`), `recursion_limit`, tool node isolation.

Build:
- `subagents/explorer.py`: A separate `StateGraph` with read-only tools (`read_file_lines`, `grep_search`) only. `recursion_limit=10`. It cannot call `apply_search_replace` or `run_command`, and it cannot spawn subagents.
- Parent graph node: `delegate_to_explorer` invokes the subgraph, receives a distilled summary (max 500 words), injects it into main state.

**Learning check:** Give the agent a task requiring 10 file reads across 3 directories. Verify the Explorer returns a summary, not raw file dumps, and that the main token budget did not grow proportionally.

**Deliverable:** CLI agent that can handle a multi-file refactor, resume after restart, and compress history.

**Success criteria:** A 15-turn session uses fewer than 60% of the model's context window; session resume restores `files_modified` exactly.

---

### Phase 3 — Production Hardening: FastAPI + HITL + Verification Hooks (Week 3)

**Goal:** Turn the CLI into a platform with an API, real-time streaming, and a web approval UI.

#### Step 3.1 — FastAPI Backend + SSE
Research first: FastAPI `StreamingResponse`, Server-Sent Events (`text/event-stream`), LangGraph streaming modes (`updates`, `messages`).

Build:
- `api/main.py`:
  - `POST /sessions` — create session, bootstrap context, return `session_id`.
  - `POST /sessions/{id}/message` — stream agent thoughts/tool calls/diffs as SSE events to the client.
  - `POST /sessions/{id}/approve` — resume a graph paused at `interrupt()`.
  - `GET /sessions/{id}/state` — inspect current `WorkingMemory` and checkpoint.

**Learning check:** Can you watch token-by-token streaming in a browser console?

#### Step 3.2 — Deterministic Verification Hook
Build into `graph.py`:
- Conditional edge after `apply_search_replace`: always run `ruff check` + project test suite inside Docker sandbox.
- Non-zero exit code routes to a `fixer` node with error traceback injected into messages. Max 3 retries, then escalate to human.

**Learning check:** Introduce a syntax error via the agent; prove the graph auto-catches it and retries.

#### Step 3.3 — Next.js Frontend (HITL)
Build:
- Live file tree (from repo snapshot).
- Side-by-side diff viewer (render `apply_search_replace` before/after).
- Terminal output stream (SSE events).
- Approval Gate modal: "Agent wants to run `pytest`. Approve / Deny / Modify command."
- Token and cost tracker (real-time counter from your instrumentation).

**Deliverable:** `docker-compose.yml` spins up backend + frontend + Postgres + Qdrant + sandbox. One URL to demo.

**Success criteria:** You can fix a bug in a sample repo entirely through the browser, with approval gates pausing execution before destructive actions.

---

### Phase 4 — Evaluation & Resume Packaging (Week 4)

**Goal:** Hard numbers for your resume and GitHub README.

#### Step 4.1 — Build the Eval Harness
Research first: SWE-bench methodology (pass@1, patch application, test execution). Pick 15–20 issues from small, well-tested Python repos (e.g., `requests`, `flask`, `httpx`, or a personal repo with known bugs).

Build:
- `evals/run_benchmark.py`:
  - Clone repo → checkout base commit → apply agent's patch → run official test suite → record pass/fail.
  - Log tokens spent per task, latency per turn, compaction triggers, approval decisions.
  - Compare two configurations: (A) naive file-read agent vs (B) your AST + compaction agent.

**Learning check:** Can you explain why configuration B uses fewer tokens but maintains the same pass rate?

#### Step 4.2 — Resume Artifacts
- **README.md:** Architecture diagram of the 6-component graph, stack badges, eval results table (e.g., "pass@1: 65% on 20 issues; 42% token reduction via AST repo-mapping").
- **90-second demo video:** Screen recording of the Next.js UI fixing a real bug with the approval gate.
- **Blog post:** "Why I chose Tree-sitter over ripgrep for code context" + "How LangGraph checkpointers saved my agent state."
- **GitHub repo:** Clean modular layout matching the structure above. No notebooks, no unused files.

---

## Decision Points to Lock Now

| Decision | Recommendation for Learning | Why |
|----------|----------------------------|-----|
| Model provider | Local Ollama with `qwen2.5-coder:7b` (or `qwen3`) | Zero API cost while iterating; same OpenAI-compatible client works with `langchain-openai`; you learn prompt-caching behavior without billing surprises |
| First practice repo | Your own small Python repo | You control the ground truth; you can inject bugs and verify fixes |
| Approval mode default | `"ask"` | Forces you to build and test the `interrupt()` HITL path early |
| Checkpointer | Start with SQLiteSaver in Phase 1, migrate to AsyncPostgresSaver in Phase 2 | You learn the abstraction before adding infrastructure complexity |

---

## Why Not Codex or OpenCode as Primary Boilerplate?

- **OpenCode:** Excellent reference for client/server decoupling, subagent/permission separation, and LSP integration. Built primarily in TypeScript/Bun. Copying it directly means fighting a TypeScript codebase rather than demonstrating deep Python systems engineering. Also carries product bloat (200+ providers, mobile clients, plugins).
- **OpenAI Codex:** Goldmine for sandboxing (Seatbell, Landlock, seccomp), git worktree parallelism, and the AGENTS.md pattern. However, its cloud harness is proprietary and tuned around OpenAI's native `apply_patch` format. It has no built-in vector/AST indexing, which weakens the systems-engineering interview story.

**Recommended hybrid:** Steal the best subsystem from each and implement cleanly in Python:
- Client-server & subagents → OpenCode patterns
- Sandboxing & repo rules → Codex patterns
- Codebase indexing & diffs → Aider patterns
- ACI tools & benchmarks → SWE-agent patterns

---

## How to Use This Plan

1. Do not write the full agent in one file. Create `src/coding_agent/` with the module structure above. Each `.py` file is one component.
2. Commit after each phase. Tag releases: `v0.1-cli-loop`, `v0.2-with-memory`, `v0.3-api`, `v1.0-evaluated`.
3. Write the ADRs. Before coding Component 3, write `docs/adr-003-permissions.md` explaining why you chose path-jail + Docker over raw shell. This is what interviewers actually ask about.
4. Instrument from Day 1. Add a token counter callback to LangChain so every LLM call logs input/output tokens. Your eval metrics depend on this.

---

## Next Immediate Step

Confirm your model provider (Ollama local vs API) and pick your first practice repository. Then begin Phase 1, Step 1.1.