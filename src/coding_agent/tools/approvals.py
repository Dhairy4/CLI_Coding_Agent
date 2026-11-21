from typing import Any
from langgraph.types import interrupt


MUTATING_TOOLS = {"apply_search_replace", "run_command"}


def check_approval(tool_name: str, tool_args: dict[str, Any], approval_mode: str) -> bool:
    """
    Check if a tool execution requires human approval.
    If approval_mode == "ask" and tool is mutating, call LangGraph interrupt().
    If approval_mode == "never", reject mutating tools.
    If approval_mode == "auto", approve automatically.
    """
    if approval_mode == "never" and tool_name in MUTATING_TOOLS:
        raise PermissionError(f"Action '{tool_name}' rejected: approval_mode is 'never'.")

    if approval_mode == "ask" and tool_name in MUTATING_TOOLS:
        # Request approval via LangGraph interrupt()
        approval_request = {
            "type": "approval_required",
            "tool": tool_name,
            "args": tool_args,
            "message": f"Agent requests permission to execute '{tool_name}' with args: {tool_args}",
        }
        user_response = interrupt(approval_request)
        # user_response can be bool (True/False) or dict {"approved": True/False}
        if isinstance(user_response, dict):
            approved = user_response.get("approved", False)
        else:
            approved = bool(user_response)

        if not approved:
            raise PermissionError(f"Action '{tool_name}' denied by user.")

    return True
