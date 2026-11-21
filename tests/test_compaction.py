from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from coding_agent.memory.compaction import (
    clip_output,
    compact_history,
    count_messages_tokens,
    count_tokens,
    dedupe_read,
    should_compact,
)


def test_clip_output():
    """Verify head/tail clipping on large tool output."""
    lines = [f"Output line {i}" for i in range(1, 101)]
    long_text = "\n".join(lines)

    clipped = clip_output(long_text, max_lines=40, head_lines=15, tail_lines=15)
    clipped_lines = clipped.splitlines()

    # Head lines present
    assert "Output line 1" in clipped_lines[0]
    assert "Output line 15" in clipped_lines[14]

    # Truncation message present
    assert any("truncated to minimize context bloat" in l for l in clipped_lines)

    # Tail lines present
    assert "Output line 100" in clipped_lines[-1]
    assert "Output line 86" in clipped_lines[-15]


def test_dedupe_read():
    """Verify SHA256 read deduplication returns pointer on unchanged files."""
    path = "src/code.py"
    file_content = "def test_func():\n    return 42\n"
    read_hashes = {}

    # First read: new file
    out1, hash1, was_deduped1 = dedupe_read(path, file_content, read_hashes)
    assert was_deduped1 is False
    assert out1 == file_content
    read_hashes[path] = hash1

    # Second read: unchanged content
    out2, hash2, was_deduped2 = dedupe_read(path, file_content, read_hashes)
    assert was_deduped2 is True
    assert hash2 == hash1
    assert "unchanged since last read" in out2

    # Third read: modified content
    new_content = "def test_func():\n    return 99\n"
    out3, hash3, was_deduped3 = dedupe_read(path, new_content, read_hashes)
    assert was_deduped3 is False
    assert hash3 != hash1
    assert out3 == new_content


def test_token_counting_and_compaction_trigger():
    """Verify token counting and compaction threshold detection."""
    short_text = "def hello(): pass"
    tokens = count_tokens(short_text)
    assert tokens > 0

    messages = [
        HumanMessage(content="A" * 1000),
        AIMessage(content="B" * 2000),
    ]
    total_tokens = count_messages_tokens(messages)
    assert total_tokens > 500

    # Under budget
    assert should_compact(messages, max_token_budget=10000, token_budget_pct=0.75) is False

    # Over budget
    assert should_compact(messages, max_token_budget=500, token_budget_pct=0.75) is True


def test_compact_history():
    """Verify rolling compaction summarizes intermediate turns."""
    messages = [
        HumanMessage(content="Initial Goal: Fix parser bug"),
        AIMessage(content="Checking syntax", tool_calls=[{"name": "read_file_lines", "args": {"path": "parser.py"}, "id": "1"}]),
        ToolMessage(content="File contents line 1-50", tool_call_id="1"),
        AIMessage(content="Found issue", tool_calls=[{"name": "read_file_lines", "args": {"path": "ast.py"}, "id": "2"}]),
        ToolMessage(content="AST definitions", tool_call_id="2"),
        AIMessage(content="Executing search", tool_calls=[{"name": "grep_search", "args": {"query": "parse"}, "id": "3"}]),
        ToolMessage(content="3 matches found", tool_call_id="3"),
        AIMessage(content="I will now apply fix"),
    ]

    compacted = compact_history(messages, goal="Fix parser bug", keep_recent=2)
    assert len(compacted) < len(messages)
    assert compacted[0].content == "Initial Goal: Fix parser bug"
    assert "Rolling Summary" in compacted[1].content
    assert compacted[-1].content == "I will now apply fix"
