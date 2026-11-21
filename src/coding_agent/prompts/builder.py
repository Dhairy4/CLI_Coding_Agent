from langchain_core.messages import AnyMessage, SystemMessage
from coding_agent.state import AgentState, WorkingMemory


STATIC_SYSTEM_PREFIX = """You are an autonomous software engineering agent built around Sebastian Raschka's 6 harness components.

Your core operating principles:
1. Live Repo Context: Always explore codebase structure and symbols before making edits.
2. Safe Deterministic Edits: Use `apply_search_replace` with exact matching unique blocks. Never guess file contents.
3. Strict Path Jailing: All file operations are confined within the workspace root.
4. Verify Everything: After proposing code changes, run tests and lint checks via the sandbox command runner.
5. Minimal Token Bloat: Keep tool calls precise, read only relevant line ranges, and rely on working memory.

Available Tools:
- read_file_lines(path, start_line, line_count): Read a slice of a file.
- grep_search(query, path_pattern, case_sensitive): Search for regex/strings across workspace files.
- search_symbols(query, limit): Semantic search for Python classes, functions, and imports in AST index.
- delegate_to_explorer(research_query): Delegate large searches or multi-file investigation to a bounded read-only subagent to preserve main context budget.
- apply_search_replace(path, old_str, new_str): Replace an EXACT unique block of code.
- run_command(command): Run a bash command inside the sandbox (pytest, ruff, git diff, etc.).

When working on a problem:
1. Read the relevant files around the target area.
2. Formulate a plan and update your working state.
3. Apply precise search/replace edits.
4. Verify by running tests/linters.
"""


def format_working_memory(memory: WorkingMemory) -> str:
    """Format dual-layer working memory into a concise markdown section."""
    sections = ["### Active Working Memory:"]
    if memory.current_goal:
        sections.append(f"**Current Goal:** {memory.current_goal}")

    if memory.plan_steps:
        sections.append("**Current Plan:**")
        for i, step in enumerate(memory.plan_steps, 1):
            sections.append(f"  {i}. {step}")

    if memory.files_modified:
        sections.append(f"**Files Modified:** {', '.join(memory.files_modified)}")

    if memory.last_verification_error:
        sections.append(f"**Last Verification Output/Error:**\n```\n{memory.last_verification_error}\n```")

    if memory.notes:
        sections.append("**Discovered Notes:**\n" + "\n".join(f"- {n}" for n in memory.notes))

    return "\n".join(sections)


def assemble_prompt_messages(state: AgentState) -> list[AnyMessage]:
    """
    Assemble messages respecting prompt cache boundaries:
    1. Static prefix (never changes per session -> cacheable prefix)
    2. Repo snapshot block (collected at session start)
    3. Working memory block (dynamically updated per turn)
    4. Conversation history (user inputs, agent tool calls, tool responses)
    """
    repo_snapshot = state.get("repo_snapshot", "")
    full_static_prefix = STATIC_SYSTEM_PREFIX
    if repo_snapshot:
        full_static_prefix += f"\n\n---\n## Repository Snapshot\n{repo_snapshot}"

    static_msg = SystemMessage(content=full_static_prefix)

    # Dynamic Working Memory block
    working_memory = state.get("working_memory")
    if isinstance(working_memory, dict):
        working_memory = WorkingMemory(**working_memory)
    elif working_memory is None:
        working_memory = WorkingMemory()

    memory_text = format_working_memory(working_memory)
    memory_msg = SystemMessage(content=memory_text)

    # Dynamic history
    history = state.get("messages", [])

    return [static_msg, memory_msg, *history]
