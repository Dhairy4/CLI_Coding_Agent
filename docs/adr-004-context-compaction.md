# ADR 004: Context Hygiene via Output Clipping, Read Dedup, and Rolling Compaction

## Status
Accepted

## Context
Long-horizon software engineering tasks quickly overflow LLM context windows or inflate inference cost. Three specific failure modes cause token bloat:
1. Massive tool outputs (e.g. running `pytest` on large suites or reading extensive logs) inject thousands of tokens in a single turn.
2. Redundant file reads when an agent repeatedly checks the same unchanged file across turns.
3. Linear growth of chat history over 15+ turns degrading model attention and increasing Time-To-First-Token (TTFT).

## Decision
We implement a three-tiered Context Hygiene pipeline in `memory/compaction.py`:
1. **Head/Tail Output Clipping (`clip_output`):** Commands and tool outputs exceeding `max_output_lines` (default: 100) are truncated by preserving the top 25 and bottom 25 lines while omitting the middle with an explicit marker: `... [N lines truncated to minimize context bloat] ...`.
2. **SHA-256 Read Deduplication (`dedupe_read`):** When reading files, the system hashes the content. If the file is unchanged since the last read, the harness returns a short pointer: `[File unchanged since last read (SHA256: ...). Referring to previous read in history.]`.
3. **Rolling Compaction (`should_compact` & `compact_history`):** Token consumption is tracked across turns using `tiktoken`. When context reaches 75% of the allocated token budget (default 8,000 tokens), older turns are compressed into a single rolling summary, preserving only the initial task and the most recent 4 working turns.

## Consequences
- **Positive:** Context token consumption remains bounded and predictable throughout extended sessions (>20 turns). Attention degradation is avoided.
- **Trade-off:** Intermediate tool outputs from early turns are compressed into summaries, though essential file changes and goals are preserved in `WorkingMemory`.
