import uuid
from pathlib import Path
from coding_agent.config import config
from coding_agent.graph import build_agent_graph
from coding_agent.memory.checkpointer import get_checkpointer
from coding_agent.state import WorkingMemory
from langchain_core.messages import HumanMessage


def test_session_resume_postgres():
    """Verify that state persists in PostgreSQL and survives across graph re-instantiation."""
    thread_id = f"test-resume-{uuid.uuid4().hex[:8]}"
    thread_config = {"configurable": {"thread_id": thread_id}}

    checkpointer = get_checkpointer()
    agent1 = build_agent_graph(checkpointer=checkpointer)

    initial_memory = WorkingMemory(
        current_goal="Refactor authentication handler",
        plan_steps=["Step 1: Inspect headers", "Step 2: Update token expiry"],
        files_modified=["src/auth.py", "src/tokens.py"],
        notes=["Found legacy jwt dependency"],
    )

    state_input = {
        "messages": [HumanMessage(content="Status check")],
        "workspace_root": str(config.workspace_root),
        "repo_snapshot": "Test snapshot",
        "approval_mode": "auto",
        "working_memory": initial_memory,
        "step_count": 0,
        "max_steps": 1,
        "pending_approval": None,
        "last_command_output": None,
    }

    # Step 1: Execute on agent instance 1
    agent1.invoke(state_input, config=thread_config)

    # Step 2: Simulate process death and reload with a brand new graph instance
    agent2 = build_agent_graph(checkpointer=checkpointer)
    restored_state = agent2.get_state(thread_config)

    assert restored_state is not None
    assert restored_state.values is not None

    restored_memory = restored_state.values.get("working_memory")
    assert restored_memory is not None

    if isinstance(restored_memory, dict):
        assert restored_memory["current_goal"] == "Refactor authentication handler"
        assert "src/auth.py" in restored_memory["files_modified"]
        assert "src/tokens.py" in restored_memory["files_modified"]
        assert "Found legacy jwt dependency" in restored_memory["notes"]
    else:
        assert restored_memory.current_goal == "Refactor authentication handler"
        assert "src/auth.py" in restored_memory.files_modified
        assert "src/tokens.py" in restored_memory.files_modified
        assert "Found legacy jwt dependency" in restored_memory.notes
