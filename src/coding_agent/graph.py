from pathlib import Path
from typing import Any, Literal
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from coding_agent.config import config
from coding_agent.context.snapshot import collect_repo_snapshot
from coding_agent.context.indexer import ASTCodeIndexer
from coding_agent.memory.checkpointer import get_checkpointer
from coding_agent.memory.compaction import clip_output, compact_history, dedupe_read, should_compact
from coding_agent.prompts.builder import assemble_prompt_messages
from coding_agent.state import AgentState, WorkingMemory
from coding_agent.subagents.explorer import delegate_to_explorer
from coding_agent.tools.filesystem import apply_search_replace, grep_search, read_file_lines
from coding_agent.tools.sandbox import run_command
from coding_agent.tools.search import search_symbols

# Master tool list
TOOLS = [
    read_file_lines,
    grep_search,
    search_symbols,
    delegate_to_explorer,
    apply_search_replace,
    run_command,
]
TOOL_MAP = {t.name: t for t in TOOLS}
MUTATING_TOOLS = {"apply_search_replace", "run_command"}


def get_model():
    """Instantiate chat model with tool bindings."""
    if config.model_provider == "nvidia":
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(
            model=config.model_name,
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key,
            temperature=config.temperature,
        )
    elif config.model_provider == "ollama":
        model = ChatOllama(
            model=config.model_name,
            base_url=config.ollama_base_url,
            temperature=config.temperature,
        )
    else:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(
            model=config.model_name,
            api_key=config.openai_api_key,
            temperature=config.temperature,
        )
    return model.bind_tools(TOOLS)


def bootstrap_context_node(state: AgentState) -> dict[str, Any]:
    """Capture live workspace repository snapshot at start."""
    workspace_path = Path(state.get("workspace_root") or config.workspace_root)
    existing_snapshot = state.get("repo_snapshot")
    if not existing_snapshot:
        snapshot = collect_repo_snapshot(workspace_path)
    else:
        snapshot = existing_snapshot

    return {"repo_snapshot": snapshot}


def agent_model_node(state: AgentState) -> dict[str, Any]:
    """Format cache-optimized prompt and call model, with context hygiene."""
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", config.max_steps)

    if step_count >= max_steps:
        return {
            "messages": [
                AIMessage(
                    content=f"Maximum iteration limit ({max_steps}) reached. Stopping to prevent infinite loops."
                )
            ],
            "step_count": step_count + 1,
        }

    # Context Hygiene: Rolling compaction when tokens exceed budget threshold
    current_msgs = state.get("messages", [])
    if should_compact(current_msgs, max_token_budget=config.token_budget, token_budget_pct=config.compaction_threshold):
        wm = state.get("working_memory")
        goal = wm.current_goal if isinstance(wm, WorkingMemory) else (wm.get("current_goal", "") if isinstance(wm, dict) else "")
        compacted_msgs = compact_history(current_msgs, goal=goal)
        state = dict(state)
        state["messages"] = compacted_msgs

    prompt_messages = assemble_prompt_messages(state)
    model = get_model()
    response = model.invoke(prompt_messages)

    return {
        "messages": [response],
        "step_count": step_count + 1,
    }


def approval_gate_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval gate for mutating actions."""
    approval_mode = state.get("approval_mode", config.approval_mode)
    last_msg = state["messages"][-1]

    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}

    mutating_calls = [
        tc for tc in last_msg.tool_calls if tc["name"] in MUTATING_TOOLS
    ]

    if approval_mode == "ask" and mutating_calls:
        # Trigger LangGraph interrupt for human authorization
        approval_request = {
            "type": "permission_request",
            "calls": mutating_calls,
            "message": f"Agent requests approval to execute: {', '.join(c['name'] for c in mutating_calls)}",
        }
        user_response = interrupt(approval_request)
        is_approved = (
            user_response.get("approved", False)
            if isinstance(user_response, dict)
            else bool(user_response)
        )
        return {"pending_approval": {"approved": is_approved, "response": user_response}}

    return {"pending_approval": {"approved": True}}


def tool_execution_node(state: AgentState) -> dict[str, Any]:
    """Execute tool calls, manage working memory updates, and wrap outputs."""
    last_msg = state["messages"][-1]
    pending = state.get("pending_approval") or {"approved": True}
    is_approved = pending.get("approved", True)

    tool_messages: list[ToolMessage] = []
    working_memory = state.get("working_memory")
    if isinstance(working_memory, dict):
        working_memory = WorkingMemory(**working_memory)
    elif working_memory is None:
        working_memory = WorkingMemory()

    for tc in last_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]
        call_id = tc["id"]

        if name in MUTATING_TOOLS and not is_approved:
            tool_messages.append(
                ToolMessage(
                    content=f"Action '{name}' was DENIED by the user.",
                    tool_call_id=call_id,
                )
            )
            continue

        selected_tool = TOOL_MAP.get(name)
        if not selected_tool:
            tool_messages.append(
                ToolMessage(
                    content=f"Error: Unknown tool '{name}'.",
                    tool_call_id=call_id,
                )
            )
            continue

        try:
            result = selected_tool.invoke(args)
            res_str = str(result)
        except Exception as e:
            res_str = f"Tool execution failed: {e}"

        # Context Hygiene: Deduplicate redundant file reads
        if name == "read_file_lines":
            target_path = args.get("path")
            if target_path and "File:" in res_str:
                res_str, file_hash, _ = dedupe_read(
                    target_path, res_str, working_memory.read_file_hashes
                )
                working_memory.read_file_hashes[target_path] = file_hash

        # Context Hygiene: Clip long outputs to minimize token bloat
        res_str = clip_output(res_str, max_lines=config.max_output_lines)

        # Update WorkingMemory for specific actions
        if name == "apply_search_replace" and "Successfully updated" in res_str:
            target_path = args.get("path")
            if target_path and target_path not in working_memory.files_modified:
                working_memory.files_modified.append(target_path)

        if name == "run_command":
            working_memory.last_verification_error = res_str[:500]

        tool_messages.append(ToolMessage(content=res_str, tool_call_id=call_id))

    return {
        "messages": tool_messages,
        "working_memory": working_memory,
        "pending_approval": None,
    }


def route_after_model(state: AgentState) -> Literal["approval_gate", "tool_executor", "__end__"]:
    """Route based on whether tool calls exist and whether approval is required."""
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", config.max_steps)

    if step_count >= max_steps:
        return END

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return END

    approval_mode = state.get("approval_mode", config.approval_mode)
    has_mutating = any(tc["name"] in MUTATING_TOOLS for tc in last_msg.tool_calls)

    if approval_mode == "ask" and has_mutating:
        return "approval_gate"
    return "tool_executor"


def build_agent_graph(checkpointer=None):
    """Construct and compile the LangGraph state machine with persistent checkpointer."""
    workflow = StateGraph(AgentState)

    workflow.add_node("bootstrap_context", bootstrap_context_node)
    workflow.add_node("agent_model", agent_model_node)
    workflow.add_node("approval_gate", approval_gate_node)
    workflow.add_node("tool_executor", tool_execution_node)

    workflow.add_edge(START, "bootstrap_context")
    workflow.add_edge("bootstrap_context", "agent_model")

    workflow.add_conditional_edges(
        "agent_model",
        route_after_model,
        {
            "approval_gate": "approval_gate",
            "tool_executor": "tool_executor",
            END: END,
        },
    )

    workflow.add_edge("approval_gate", "tool_executor")
    workflow.add_edge("tool_executor", "agent_model")

    active_checkpointer = checkpointer if checkpointer is not None else get_checkpointer()
    return workflow.compile(checkpointer=active_checkpointer)
