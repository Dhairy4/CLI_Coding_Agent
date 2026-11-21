from coding_agent.subagents.explorer import (
    build_explorer_subgraph,
    delegate_to_explorer,
    run_explorer,
    EXPLORER_TOOLS,
)


def test_explorer_tool_isolation():
    """Verify Explorer subagent is isolated to read-only tools."""
    tool_names = [t.name for t in EXPLORER_TOOLS]
    assert "read_file_lines" in tool_names
    assert "grep_search" in tool_names
    assert "search_symbols" in tool_names

    # Crucial security check: No mutating or shell tools allowed in Explorer
    assert "apply_search_replace" not in tool_names
    assert "run_command" not in tool_names
    assert "delegate_to_explorer" not in tool_names


def test_explorer_delegation():
    """Verify delegate_to_explorer tool invocation returns a concise brief."""
    result = delegate_to_explorer.invoke(
        {"research_query": "Find where AgentConfig is defined in src/coding_agent/config.py"}
    )

    assert isinstance(result, str)
    assert len(result) > 0
    # Must contain subagent findings heading
    assert "Explorer Subagent Findings" in result
