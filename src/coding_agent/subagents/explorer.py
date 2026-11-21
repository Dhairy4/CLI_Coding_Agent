from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from coding_agent.config import config
from coding_agent.tools.filesystem import grep_search, read_file_lines
from coding_agent.tools.search import search_symbols

# Read-only tools strictly allowed for Explorer (NO write or shell tools allowed)
EXPLORER_TOOLS = [read_file_lines, grep_search, search_symbols]
EXPLORER_TOOL_MAP = {t.name: t for t in EXPLORER_TOOLS}

EXPLORER_SYSTEM_PROMPT = """You are an Explorer Subagent for an autonomous software engineering platform.
Your ONLY responsibility is to read, grep, and analyze code to answer a specific research query.

STRICT CONSTRAINTS:
1. You have ONLY read-only tools: `read_file_lines`, `grep_search`, `search_symbols`.
2. You CANNOT edit files, run shell commands, or spawn subagents.
3. Keep tool calls focused. Do not read entire files if reading a 50-line range suffices.
4. When finished, provide a concise, distilled technical summary (maximum 300-400 words) detailing:
   - Key file paths and line ranges
   - Symbol definitions and patterns found
   - Actionable conclusions for the primary engineering agent.
"""


class ExplorerState(TypedDict):
    """Isolated state for the bounded Explorer subagent."""
    messages: Annotated[list[AnyMessage], add_messages]
    task: str
    step_count: int
    max_steps: int
    final_summary: str | None


def get_explorer_model():
    """Instantiate model for Explorer subgraph."""
    if config.model_provider == "nvidia":
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(
            model=config.model_name,
            base_url=config.nvidia_base_url,
            api_key=config.nvidia_api_key,
            temperature=0.0,
        )
    elif config.model_provider == "ollama":
        from langchain_ollama import ChatOllama
        model = ChatOllama(
            model=config.model_name,
            base_url=config.ollama_base_url,
            temperature=0.0,
        )
    else:
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(
            model=config.model_name,
            api_key=config.openai_api_key,
            temperature=0.0,
        )
    return model.bind_tools(EXPLORER_TOOLS)


def explorer_model_node(state: ExplorerState) -> dict[str, Any]:
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 6)

    if step_count >= max_steps:
        return {
            "messages": [
                AIMessage(
                    content="Explorer reached maximum step limit. Summarizing findings collected so far."
                )
            ],
            "step_count": step_count + 1,
            "final_summary": "Explorer stopped at max step limit.",
        }

    model = get_explorer_model()
    # Ensure system prompt is present
    msgs = state.get("messages", [])
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=EXPLORER_SYSTEM_PROMPT), *msgs]

    response = model.invoke(msgs)
    summary = None
    if not response.tool_calls and response.content:
        summary = str(response.content)

    return {
        "messages": [response],
        "step_count": step_count + 1,
        "final_summary": summary,
    }


def explorer_tool_node(state: ExplorerState) -> dict[str, Any]:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}

    tool_messages: list[ToolMessage] = []
    for tc in last_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]
        call_id = tc["id"]

        if name not in EXPLORER_TOOL_MAP:
            tool_messages.append(
                ToolMessage(
                    content=f"Error: Explorer cannot invoke unauthorized tool '{name}'.",
                    tool_call_id=call_id,
                )
            )
            continue

        selected = EXPLORER_TOOL_MAP[name]
        try:
            res = str(selected.invoke(args))
        except Exception as e:
            res = f"Tool error: {e}"

        tool_messages.append(ToolMessage(content=res, tool_call_id=call_id))

    return {"messages": tool_messages}


def route_explorer(state: ExplorerState) -> Literal["explorer_tools", "__end__"]:
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 6)
    if step_count >= max_steps:
        return END

    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return END

    return "explorer_tools"


def build_explorer_subgraph():
    """Build bounded read-only Explorer subgraph."""
    graph = StateGraph(ExplorerState)

    graph.add_node("explorer_model", explorer_model_node)
    graph.add_node("explorer_tools", explorer_tool_node)

    graph.add_edge(START, "explorer_model")
    graph.add_conditional_edges(
        "explorer_model",
        route_explorer,
        {"explorer_tools": "explorer_tools", END: END},
    )
    graph.add_edge("explorer_tools", "explorer_model")

    return graph.compile()


_EXPLORER_SUBGRAPH = None


def get_explorer_subgraph():
    global _EXPLORER_SUBGRAPH
    if _EXPLORER_SUBGRAPH is None:
        _EXPLORER_SUBGRAPH = build_explorer_subgraph()
    return _EXPLORER_SUBGRAPH


def run_explorer(task_description: str, max_steps: int = 6) -> str:
    """Execute bounded Explorer subgraph and return distilled summary."""
    subgraph = get_explorer_subgraph()
    initial_state: ExplorerState = {
        "messages": [
            SystemMessage(content=EXPLORER_SYSTEM_PROMPT),
            HumanMessage(content=f"Investigate codebase for: {task_description}"),
        ],
        "task": task_description,
        "step_count": 0,
        "max_steps": max_steps,
        "final_summary": None,
    }

    final_state = subgraph.invoke(initial_state, {"recursion_limit": 10})
    if final_state.get("final_summary"):
        return final_state["final_summary"]

    # Extract latest AIMessage content
    for msg in reversed(final_state.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)

    return "Explorer completed analysis but produced no textual summary."


@tool
def delegate_to_explorer(research_query: str) -> str:
    """Delegate a multi-file read/search codebase investigation to the bounded Explorer subagent.

    The Explorer uses isolated read-only tools (grep, AST symbols, line slices) and returns
    a distilled technical summary, avoiding bloating your main context window with raw dumps.

    Args:
        research_query: What files, symbols, patterns, or architecture to investigate.
    """
    try:
        summary = run_explorer(research_query, max_steps=5)
        return f"[Explorer Subagent Findings for: '{research_query}']:\n{summary}"
    except Exception as e:
        return f"Explorer subagent delegation failed: {e}"
