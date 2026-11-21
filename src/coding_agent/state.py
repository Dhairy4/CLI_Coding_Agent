from typing import Annotated, Any, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class WorkingMemory(BaseModel):
    """Dual-layer compact working memory updated across turns."""
    current_goal: str = Field(default="", description="The primary goal or user task.")
    plan_steps: list[str] = Field(default_factory=list, description="Planned action steps.")
    files_modified: list[str] = Field(default_factory=list, description="Files modified during this session.")
    read_file_hashes: dict[str, str] = Field(default_factory=dict, description="Path to sha256 hash of read files.")
    last_verification_error: str | None = Field(default=None, description="Output/error from last test/lint execution.")
    notes: list[str] = Field(default_factory=list, description="Key insights or facts discovered.")


class AgentState(TypedDict):
    """LangGraph State holding conversation history and harness metadata."""
    messages: Annotated[list[AnyMessage], add_messages]
    workspace_root: str
    repo_snapshot: str
    approval_mode: Literal["ask", "auto", "never"]
    working_memory: WorkingMemory
    step_count: int
    max_steps: int
    pending_approval: dict[str, Any] | None
    last_command_output: str | None
