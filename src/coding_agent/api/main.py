import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from coding_agent.config import config
from coding_agent.context.snapshot import collect_repo_snapshot
from coding_agent.graph import build_agent_graph
from coding_agent.memory.compaction import count_messages_tokens
from coding_agent.state import WorkingMemory

app = FastAPI(
    title="Autonomous Coding Agent Platform API",
    description="Production-grade API with SSE streaming, deterministic verification, and HITL approval gates.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active compiled agent graph
_AGENT = build_agent_graph()

# Request/Response Schemas
class CreateSessionRequest(BaseModel):
    workspace_root: str | None = None
    approval_mode: str | None = None


class SendMessageRequest(BaseModel):
    message: str = Field(..., description="User prompt or instruction for coding agent.")


class ApproveRequest(BaseModel):
    approved: bool = Field(..., description="Whether to approve or reject the action.")


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "provider": config.model_provider,
        "model": config.model_name,
        "workspace": str(config.workspace_root),
        "approval_mode": config.approval_mode,
        "checkpointer": "PostgresSaver" if config.use_postgres_checkpointer else "MemorySaver",
    }


@app.post("/sessions")
def create_session(req: CreateSessionRequest):
    session_id = f"session-{uuid.uuid4().hex[:12]}"
    workspace = req.workspace_root or str(config.workspace_root)
    snapshot = collect_repo_snapshot(Path(workspace))

    initial_state = {
        "messages": [],
        "workspace_root": workspace,
        "repo_snapshot": snapshot,
        "approval_mode": req.approval_mode or config.approval_mode,
        "working_memory": WorkingMemory(),
        "step_count": 0,
        "max_steps": config.max_steps,
        "pending_approval": None,
        "last_command_output": None,
        "verification_retries": 0,
        "last_verification_passed": None,
    }

    graph_config = {"configurable": {"thread_id": session_id}}
    _AGENT.update_state(graph_config, initial_state)

    return {
        "session_id": session_id,
        "workspace_root": workspace,
        "repo_snapshot": snapshot,
    }


@app.get("/sessions/{session_id}/state")
def get_session_state(session_id: str):
    graph_config = {"configurable": {"thread_id": session_id}}
    state_tuple = _AGENT.get_state(graph_config)
    if not state_tuple or not state_tuple.values:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    values = state_tuple.values
    wm = values.get("working_memory")
    wm_dict = wm.model_dump() if isinstance(wm, WorkingMemory) else (wm or {})

    messages_summary = []
    for m in values.get("messages", []):
        messages_summary.append({
            "type": getattr(m, "type", "message"),
            "content": m.content if isinstance(m.content, str) else str(m.content),
            "tool_calls": getattr(m, "tool_calls", []),
        })

    total_tokens = count_messages_tokens(values.get("messages", []))

    return {
        "session_id": session_id,
        "step_count": values.get("step_count", 0),
        "approval_mode": values.get("approval_mode", config.approval_mode),
        "working_memory": wm_dict,
        "pending_approval": values.get("pending_approval"),
        "last_verification_passed": values.get("last_verification_passed"),
        "total_tokens_spent": total_tokens,
        "message_count": len(messages_summary),
        "messages": messages_summary,
    }


@app.get("/sessions/{session_id}/snapshot")
def get_snapshot(session_id: str):
    graph_config = {"configurable": {"thread_id": session_id}}
    state_tuple = _AGENT.get_state(graph_config)
    if not state_tuple or not state_tuple.values:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"repo_snapshot": state_tuple.values.get("repo_snapshot", "")}


async def event_generator(stream_input: Any, graph_config: dict[str, Any]) -> AsyncGenerator[dict[str, str], None]:
    """Stream LangGraph state machine events as Server-Sent Events."""
    interrupted = False
    interrupt_payload = None

    try:
        for event in _AGENT.stream(stream_input, graph_config, stream_mode="updates"):
            if "__interrupt__" in event:
                interrupted = True
                interrupt_payload = event["__interrupt__"][0].value
                yield {
                    "event": "approval_required",
                    "data": json.dumps(interrupt_payload),
                }
                break

            for node_name, updates in event.items():
                # Notify node transition
                yield {
                    "event": "node_update",
                    "data": json.dumps({"node": node_name}),
                }

                # Stream messages
                for msg in updates.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if msg.content:
                            yield {
                                "event": "thought",
                                "data": json.dumps({"text": msg.content}),
                            }
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                yield {
                                    "event": "tool_call",
                                    "data": json.dumps({
                                        "name": tc["name"],
                                        "args": tc["args"],
                                        "id": tc.get("id"),
                                    }),
                                }
                    elif isinstance(msg, ToolMessage):
                        yield {
                            "event": "tool_result",
                            "data": json.dumps({
                                "content": msg.content,
                                "tool_call_id": msg.tool_call_id,
                            }),
                        }

                if "last_verification_passed" in updates:
                    yield {
                        "event": "verification",
                        "data": json.dumps({
                            "passed": updates["last_verification_passed"],
                            "retries": updates.get("verification_retries", 0),
                        }),
                    }

        if not interrupted:
            yield {
                "event": "done",
                "data": json.dumps({"status": "completed"}),
            }

    except Exception as e:
        yield {
            "event": "error",
            "data": json.dumps({"error": str(e)}),
        }


@app.post("/sessions/{session_id}/message")
async def post_message(session_id: str, req: SendMessageRequest):
    graph_config = {"configurable": {"thread_id": session_id}}
    state_tuple = _AGENT.get_state(graph_config)
    if not state_tuple or not state_tuple.values:
        create_session(CreateSessionRequest())

    stream_input = {"messages": [HumanMessage(content=req.message)]}
    return EventSourceResponse(event_generator(stream_input, graph_config))


@app.post("/sessions/{session_id}/approve")
async def approve_action(session_id: str, req: ApproveRequest):
    graph_config = {"configurable": {"thread_id": session_id}}
    stream_input = Command(resume={"approved": req.approved})
    return EventSourceResponse(event_generator(stream_input, graph_config))


# Serve Web Dashboard
@app.get("/", response_class=HTMLResponse)
def index_page():
    html_file = Path(__file__).parent / "dashboard.html"
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Autonomous Coding Agent API</h1><p>Visit /health or /docs</p>")
