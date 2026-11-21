import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file
load_dotenv()


class AgentConfig(BaseSettings):
    """Configuration for the autonomous coding agent harness."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    # Workspace directory jail
    workspace_root: Path = Path(".").resolve()

    # LLM Settings (defaults to NVIDIA NIM Nemotron)
    model_provider: Literal["nvidia", "ollama", "openai"] = Field(
        default_factory=lambda: os.getenv("AGENT_MODEL_PROVIDER", "nvidia")
    )
    model_name: str = Field(
        default_factory=lambda: os.getenv("AGENT_MODEL_NAME", "nvidia/nemotron-3.5-super-vl-120b-a12b")
    )
    temperature: float = 0.0

    # NVIDIA NIM Settings
    nvidia_api_key: str | None = Field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY") or os.getenv("AGENT_NVIDIA_API_KEY")
    )
    nvidia_base_url: str = Field(
        default_factory=lambda: os.getenv("AGENT_NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    )

    # Ollama Settings
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI Settings
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    # Token and Step Limits
    max_steps: int = 25
    max_file_lines_read: int = 250
    max_output_lines: int = 100
    token_budget: int = 8000
    compaction_threshold: float = 0.75

    # Permissions and Approval Mode: "ask" | "auto" | "never"
    approval_mode: Literal["ask", "auto", "never"] = Field(
        default_factory=lambda: os.getenv("AGENT_APPROVAL_MODE", "ask")
    )

    # Checkpointer & Persistence (PostgreSQL)
    postgres_uri: str = Field(
        default_factory=lambda: os.getenv(
            "POSTGRES_URI", "postgresql://dhairya:dhairya@localhost:5432/postgres"
        )
    )
    use_postgres_checkpointer: bool = Field(
        default_factory=lambda: os.getenv("USE_POSTGRES_CHECKPOINTER", "true").lower() == "true"
    )

    # Docker Sandbox Settings
    sandbox_image: str = "coding-agent-sandbox:latest"
    sandbox_timeout_seconds: int = 60
    sandbox_mem_limit: str = "512m"
    use_docker_sandbox: bool = True

    # Vector store (Qdrant)
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "code_symbols"
    embedding_model: str = "nomic-embed-text:latest"


config = AgentConfig()
