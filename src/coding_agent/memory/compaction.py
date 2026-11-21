import hashlib
from typing import Any
import tiktoken
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage, SystemMessage

try:
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENCODER = None


def count_tokens(text: str) -> int:
    """Estimate token count using tiktoken cl100k_base or char heuristic."""
    if not text:
        return 0
    if _TIKTOKEN_ENCODER is not None:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4)


def count_messages_tokens(messages: list[AnyMessage]) -> int:
    """Calculate aggregate token consumption across all conversation turns."""
    total = 0
    for m in messages:
        if isinstance(m.content, str):
            total += count_tokens(m.content)
        elif isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, str):
                    total += count_tokens(part)
                elif isinstance(part, dict) and "text" in part:
                    total += count_tokens(part["text"])

        # Count tokens in tool calls
        if hasattr(m, "tool_calls") and m.tool_calls:
            total += count_tokens(str(m.tool_calls))
    return total


def clip_output(text: str, max_lines: int = 60, head_lines: int = 25, tail_lines: int = 25) -> str:
    """
    Clip long command or tool outputs to minimize context bloat (Component 4).
    Keeps head and tail lines while dropping intermediate lines.
    """
    if not text:
        return ""

    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    head = lines[:head_lines]
    tail = lines[-tail_lines:] if tail_lines > 0 else []
    truncated_count = len(lines) - (head_lines + tail_lines)

    result = list(head)
    result.append(f"\n... [{truncated_count} lines truncated to minimize context bloat] ...\n")
    result.extend(tail)

    return "\n".join(result)


def dedupe_read(path: str, content: str, read_hashes: dict[str, str]) -> tuple[str, str, bool]:
    """
    Deduplicate file content against previous reads via sha256 hash (Component 4).
    Returns (output_content, file_hash, was_deduped).
    """
    file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if path in read_hashes and read_hashes[path] == file_hash:
        pointer_msg = (
            f"[File '{path}' unchanged since last read (SHA256: {file_hash[:8]}). "
            f"Content identical to previous read in history.]"
        )
        return pointer_msg, file_hash, True

    return content, file_hash, False


def should_compact(
    messages: list[AnyMessage],
    max_token_budget: int = 8000,
    token_budget_pct: float = 0.75,
) -> bool:
    """Check if message tokens exceed compaction threshold (e.g. 75% of budget)."""
    current_tokens = count_messages_tokens(messages)
    threshold = int(max_token_budget * token_budget_pct)
    return current_tokens >= threshold


def compact_history(
    messages: list[AnyMessage],
    goal: str = "",
    model: Any = None,
    keep_recent: int = 4,
) -> list[AnyMessage]:
    """
    Compress older message history into a concise summary while preserving recent context.
    - Preserves messages[0] (original task).
    - Preserves last `keep_recent` messages (active working window).
    - Summarizes intermediate messages.
    """
    if len(messages) <= keep_recent + 1:
        return messages

    initial_msg = messages[0]
    middle_msgs = messages[1:-keep_recent]
    recent_msgs = messages[-keep_recent:]

    # Build middle summary
    summary_text = ""
    if model is not None:
        try:
            middle_repr = []
            for m in middle_msgs:
                role = getattr(m, "type", "message")
                content = m.content[:300] if isinstance(m.content, str) else str(m.content)[:300]
                middle_repr.append(f"{role}: {content}")
            transcript_slice = "\n".join(middle_repr)

            prompt = (
                f"You are a context compaction engine for a coding agent.\n"
                f"Current Goal: {goal}\n"
                f"Summarize the key files inspected, modifications made, errors encountered, "
                f"and progress achieved in these {len(middle_msgs)} turns. Keep it under 150 words:\n\n"
                f"{transcript_slice}"
            )
            resp = model.invoke([HumanMessage(content=prompt)])
            summary_text = resp.content.strip() if isinstance(resp.content, str) else str(resp.content)
        except Exception:
            summary_text = ""

    if not summary_text:
        # Deterministic fallback summary
        touched_files = set()
        tools_called = []
        for m in middle_msgs:
            if hasattr(m, "tool_calls") and m.tool_calls:
                for tc in m.tool_calls:
                    tools_called.append(tc.get("name", "tool"))
                    if "path" in tc.get("args", {}):
                        touched_files.add(tc["args"]["path"])

        summary_text = (
            f"Compacted {len(middle_msgs)} intermediate turns. "
            f"Actions performed: {', '.join(set(tools_called)) if tools_called else 'general deliberation'}. "
            f"Files referenced: {', '.join(touched_files) if touched_files else 'none'}."
        )

    summary_msg = AIMessage(
        content=(
            f"### [Context Compaction — Rolling Summary of {len(middle_msgs)} Prior Turns]\n"
            f"{summary_text}"
        )
    )

    return [initial_msg, summary_msg, *recent_msgs]
