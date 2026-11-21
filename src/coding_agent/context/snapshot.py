import subprocess
from pathlib import Path

DEFAULT_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "dist",
    "build",
    ".egg-info",
    ".idea",
    ".vscode",
}

DEFAULT_INSTRUCTION_FILES = [
    "AGENTS.md",
    "CLAUDE.md",
    "AGENT.md",
    ".cursorrules",
]


def get_git_info(workspace_root: Path) -> dict[str, str]:
    """Retrieve git branch and short status if inside a git repository."""
    info = {"branch": "unknown", "status": "not a git repository"}
    try:
        branch_res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if branch_res.returncode == 0:
            info["branch"] = branch_res.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--short"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if status_res.returncode == 0:
            status_text = status_res.stdout.strip()
            info["status"] = status_text if status_text else "working tree clean"
    except Exception as e:
        info["status"] = f"git check failed: {e}"

    return info


def get_repo_instructions(workspace_root: Path, max_lines: int = 150) -> dict[str, str]:
    """Collect guidance from AGENTS.md, CLAUDE.md, etc."""
    instructions: dict[str, str] = {}
    for filename in DEFAULT_INSTRUCTION_FILES:
        target = workspace_root / filename
        if target.is_file():
            try:
                lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
                truncated = "\n".join(lines[:max_lines])
                if len(lines) > max_lines:
                    truncated += f"\n... [truncated {len(lines) - max_lines} lines]"
                instructions[filename] = truncated
            except Exception as e:
                instructions[filename] = f"Error reading: {e}"
    return instructions


def build_directory_tree(
    root_path: Path,
    max_depth: int = 4,
    max_entries: int = 200,
    ignored_dirs: set[str] | None = None,
) -> str:
    """Generate a clean, bounded ASCII tree of workspace directory."""
    if ignored_dirs is None:
        ignored_dirs = DEFAULT_IGNORED_DIRS

    lines: list[str] = [root_path.name + "/"]
    count = 0

    def walk(directory: Path, prefix: str, depth: int):
        nonlocal count
        if depth > max_depth or count >= max_entries:
            return

        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            lines.append(f"{prefix}└── [Permission Denied]")
            return

        filtered_entries = [
            e for e in entries if not (e.name in ignored_dirs or e.name.startswith("."))
        ]

        total = len(filtered_entries)
        for i, entry in enumerate(filtered_entries):
            if count >= max_entries:
                lines.append(f"{prefix}... [truncated further entries]")
                break
            count += 1
            is_last = i == total - 1
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                walk(entry, prefix + child_prefix, depth + 1)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")

    walk(root_path, "", 1)
    return "\n".join(lines)


def collect_repo_snapshot(
    workspace_root: Path,
    max_depth: int = 4,
    max_tree_entries: int = 200,
) -> str:
    """Assemble live repo context: git state, instructions, and bounded tree."""
    resolved_root = workspace_root.resolve()
    git_info = get_git_info(resolved_root)
    instructions = get_repo_instructions(resolved_root)
    tree_str = build_directory_tree(resolved_root, max_depth=max_depth, max_entries=max_tree_entries)

    sections = [
        f"### Workspace Root: `{resolved_root}`",
        f"**Git Branch:** `{git_info['branch']}`",
        f"**Git Status:**\n```\n{git_info['status']}\n```",
    ]

    if instructions:
        sections.append("### Repo Instructions:")
        for filename, content in instructions.items():
            sections.append(f"#### `{filename}`\n```markdown\n{content}\n```")

    sections.append("### Directory Structure:")
    sections.append(f"```\n{tree_str}\n```")

    return "\n\n".join(sections)


if __name__ == "__main__":
    print(collect_repo_snapshot(Path(".")))
