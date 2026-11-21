from pathlib import Path
from langchain_core.messages import HumanMessage
from coding_agent.config import config
from coding_agent.graph import build_agent_graph
from coding_agent.state import WorkingMemory


def test_graph_bootstrap_and_run(tmp_path: Path):
    """Test full LangGraph flow in 'auto' mode."""
    # Setup temporary file
    sample = tmp_path / "calc.py"
    sample.write_text("def add(a, b):\n    return a - b  # bug\n", encoding="utf-8")

    orig_root = config.workspace_root
    orig_mode = config.approval_mode
    config.workspace_root = tmp_path
    config.approval_mode = "auto"

    try:
        agent = build_agent_graph()
        thread_config = {"configurable": {"thread_id": "test-thread-1"}}

        state_input = {
            "messages": [HumanMessage(content="What functions exist in calc.py? Read calc.py.")],
            "workspace_root": str(tmp_path),
            "repo_snapshot": "",
            "approval_mode": "auto",
            "working_memory": WorkingMemory(current_goal="Inspect calc.py"),
            "step_count": 0,
            "max_steps": 5,
            "pending_approval": None,
            "last_command_output": None,
        }

        # Run invocation
        final_state = agent.invoke(state_input, config=thread_config)

        assert final_state["repo_snapshot"] != ""
        assert len(final_state["messages"]) >= 2
        assert any("calc.py" in str(m.content) for m in final_state["messages"])

    finally:
        config.workspace_root = orig_root
        config.approval_mode = orig_mode
