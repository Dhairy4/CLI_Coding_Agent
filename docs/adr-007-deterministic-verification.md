# ADR 007: Deterministic Verification Hooks & Automated Fix Loops

## Status
Accepted

## Context
When an LLM coding agent applies code modifications (`apply_search_replace`), there is always a non-zero probability that the edit introduces syntax errors, indentation mismatches, or regression breaks. If unchecked, the agent drifts further from the goal by stacking hallucinations on top of broken syntax.

## Decision
We enforce a mandatory deterministic verification hook directly in the LangGraph state machine:
1. **Verification Node (`verify_node`):**
   - Automatically triggered immediately following any successful code mutation (`apply_search_replace`).
   - Executes `ruff check .` inside the isolated Docker sandbox.
2. **Auto-Fixer Loop (`fixer_node`):**
   - If the linter reports errors (non-zero exit code or syntax failures), the exact compiler/linter traceback is captured into `WorkingMemory.last_verification_error`.
   - The graph transitions to `fixer_node`, which injects the traceback into the model's message history and prompts it to resolve the error.
   - Bounded to a maximum of 3 automated retry attempts. If unresolved after 3 attempts, the graph stops and escalates to human intervention via LangGraph `interrupt()`.
3. **Clean Promotion:**
   - If verification passes cleanly, `WorkingMemory.last_verification_error` is cleared, `last_verification_passed` is set to `True`, and execution continues normally.

## Consequences
- **Positive:** Zero unverified code escapes the agent turn. Syntax errors are caught and corrected deterministically before the user even reviews the diff.
- **Trade-off:** Adds a quick lint check cycle after each edit, easily offset by preventing compounding syntax failures.
