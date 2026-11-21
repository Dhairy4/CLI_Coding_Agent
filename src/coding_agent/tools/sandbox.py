import subprocess
from pathlib import Path
from langchain_core.tools import tool
import docker

from coding_agent.config import config


def run_local_fallback(command: str, workspace_root: Path, timeout: int = 60) -> str:
    """Run command locally as fallback if Docker daemon is not active."""
    try:
        res = subprocess.run(
            ["bash", "-c", command],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = res.stdout.strip()
        stderr = res.stderr.strip()
        code = res.returncode
        out = f"Exit code: {code}\n"
        if stdout:
            out += f"STDOUT:\n{stdout}\n"
        if stderr:
            out += f"STDERR:\n{stderr}\n"
        return out.strip()
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing local command: {e}"


@tool
def run_command(command: str) -> str:
    """Execute a shell command inside an isolated Docker sandbox container.

    Use for running pytest, ruff, git status, git diff, etc.

    Args:
        command: The shell command to run.
    """
    workspace = config.workspace_root.resolve()
    if not config.use_docker_sandbox:
        return run_local_fallback(command, workspace, config.sandbox_timeout_seconds)

    try:
        client = docker.from_env()
        # Verify docker image exists
        try:
            client.images.get(config.sandbox_image)
        except docker.errors.ImageNotFound:
            # Build sandbox image if missing
            dockerfile_path = workspace / "docker"
            if (dockerfile_path / "Dockerfile.sandbox").exists():
                client.images.build(
                    path=str(dockerfile_path),
                    dockerfile="Dockerfile.sandbox",
                    tag=config.sandbox_image,
                    rm=True,
                )
            else:
                return run_local_fallback(command, workspace, config.sandbox_timeout_seconds)

        # Run container ephemeral with network disabled and workspace mounted
        container = client.containers.run(
            image=config.sandbox_image,
            command=["bash", "-c", command],
            network_mode="none",
            mem_limit=config.sandbox_mem_limit,
            volumes={
                str(workspace): {"bind": "/workspace", "mode": "rw"}
            },
            working_dir="/workspace",
            detach=False,
            stdout=True,
            stderr=True,
            remove=True,
        )
        return container.decode("utf-8", errors="replace")

    except docker.errors.ContainerError as e:
        return f"Container exit code {e.exit_status}:\n{e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)}"
    except Exception as e:
        # Fallback to local execution if Docker fails or has permission issues
        return f"[Docker sandbox unavailable ({e}); running local fallback]\n" + run_local_fallback(
            command, workspace, config.sandbox_timeout_seconds
        )
