from pathlib import Path
from starlette.testclient import TestClient
from coding_agent.api.main import app
from coding_agent.config import config
from coding_agent.graph import verify_node, route_after_tool, route_after_verify
from coding_agent.state import WorkingMemory


def test_api_health_and_session_lifecycle():
    """Verify Phase 3 FastAPI endpoints for session creation and inspection."""
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert health_data["status"] == "online"
    assert "model" in health_data

    # 2. Create session
    res_create = client.post("/sessions", json={"approval_mode": "auto"})
    assert res_create.status_code == 200
    create_data = res_create.json()
    assert "session_id" in create_data
    assert "repo_snapshot" in create_data
    session_id = create_data["session_id"]

    # 3. Inspect session state
    res_state = client.get(f"/sessions/{session_id}/state")
    assert res_state.status_code == 200
    state_data = res_state.json()
    assert state_data["session_id"] == session_id
    assert "working_memory" in state_data
    assert "total_tokens_spent" in state_data

    # 4. Fetch snapshot
    res_snap = client.get(f"/sessions/{session_id}/snapshot")
    assert res_snap.status_code == 200
    assert "repo_snapshot" in res_snap.json()


def test_deterministic_verification_routing():
    """Verify conditional edge routing after tool execution and verification."""
    # Tool execution with edits routes to verify_node
    state_with_edits = {"last_command_output": "had_edits"}
    assert route_after_tool(state_with_edits) == "verify_node"

    # Tool execution with no edits routes back to agent
    state_no_edits = {"last_command_output": None}
    assert route_after_tool(state_no_edits) == "agent_model"

    # Verification passed routes to agent_model
    state_passed = {"last_verification_passed": True, "verification_retries": 0}
    assert route_after_verify(state_passed) == "agent_model"

    # Verification failed with < 3 retries routes to fixer_node
    state_failed_retry = {"last_verification_passed": False, "verification_retries": 1}
    assert route_after_verify(state_failed_retry) == "fixer_node"

    # Verification failed with > 3 retries escalates to END
    state_exhausted = {"last_verification_passed": False, "verification_retries": 4}
    assert route_after_verify(state_exhausted) == "__end__"


def test_verify_node_execution(tmp_path: Path):
    """Verify verify_node runs check and reports clean pass on clean code."""
    orig_root = config.workspace_root
    config.workspace_root = tmp_path

    try:
        clean_file = tmp_path / "valid.py"
        clean_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

        state = {
            "messages": [],
            "workspace_root": str(tmp_path),
            "repo_snapshot": "",
            "approval_mode": "auto",
            "working_memory": WorkingMemory(),
            "step_count": 0,
            "max_steps": 5,
            "pending_approval": None,
            "last_command_output": "had_edits",
            "verification_retries": 0,
            "last_verification_passed": None,
        }

        result = verify_node(state)
        assert result["last_verification_passed"] is True
        assert result["verification_retries"] == 0
        assert "cleanly" in result["messages"][0].content

    finally:
        config.workspace_root = orig_root
