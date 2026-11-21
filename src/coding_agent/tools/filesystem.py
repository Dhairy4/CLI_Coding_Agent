import fnmatch
import os
import re
from pathlib import Path
from langchain_core.tools import tool
from coding_agent.config import config


def resolve_safe_path(rel_or_abs_path: str, workspace_root: Path | None = None) -> Path:
    """Resolve path against workspace root and jail within it."""
    root = (workspace_root or config.workspace_root).resolve()
    target = (root / rel_or_abs_path).resolve()

    # Path traversal check
    try:
        target.relative_to(root)
    except ValueError:
        raise PermissionError(f"Access denied: Path '{rel_or_abs_path}' escapes workspace jail '{root}'.")

    return target


@tool
def read_file_lines(path: str, start_line: int = 1, line_count: int = 100) -> str:
    """Read a specific range of lines from a workspace file (1-indexed).

    Args:
        path: Relative path to the file inside workspace.
        start_line: Starting line number (1-based index).
        line_count: Maximum number of lines to read.
    """
    try:
        resolved = resolve_safe_path(path)
        if not resolved.exists():
            return f"Error: File not found: {path}"
        if resolved.is_dir():
            return f"Error: {path} is a directory, not a file."

        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        total_lines = len(lines)

        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, start_idx + max(1, line_count))

        selected = lines[start_idx:end_idx]
        output = [f"File: {path} (lines {start_idx + 1}-{end_idx} of {total_lines})"]
        for idx, line in enumerate(selected, start=start_idx + 1):
            output.append(f"{idx:4d} | {line}")

        return "\n".join(output)
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def grep_search(query: str, path_pattern: str = "*", case_sensitive: bool = False) -> str:
    """Search for regex or substring pattern across workspace files.

    Args:
        query: Regex or substring pattern to search for.
        path_pattern: Glob pattern for filtering file names (e.g. '*.py').
        case_sensitive: Whether the search is case-sensitive.
    """
    try:
        root = config.workspace_root.resolve()
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(query, flags)

        matches: list[str] = []
        max_results = 50
        count = 0

        ignored = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build"}

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".")]

            for filename in filenames:
                if filename.startswith("."):
                    continue
                if path_pattern != "*" and not fnmatch.fnmatch(filename, path_pattern):
                    continue

                filepath = Path(dirpath) / filename
                rel_path = filepath.relative_to(root)

                try:
                    content = filepath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                for line_idx, line in enumerate(content.splitlines(), 1):
                    if pattern.search(line):
                        matches.append(f"{rel_path}:{line_idx}: {line.strip()}")
                        count += 1
                        if count >= max_results:
                            matches.append(f"... truncated at {max_results} matches")
                            return "\n".join(matches)

        if not matches:
            return f"No matches found for query '{query}'."
        return "\n".join(matches)
    except Exception as e:
        return f"Error in grep_search: {e}"


@tool
def apply_search_replace(path: str, old_str: str, new_str: str) -> str:
    """Apply an exact, unique search and replace edit to a file.

    Fails if old_str does not match or matches multiple times to guarantee safety.

    Args:
        path: Relative path to file.
        old_str: The exact block of text to be replaced (must be unique in file).
        new_str: The replacement text.
    """
    try:
        resolved = resolve_safe_path(path)
        if not resolved.exists():
            return f"Error: File not found: {path}"

        content = resolved.read_text(encoding="utf-8")
        count = content.count(old_str)

        if count == 0:
            return (
                f"Error: Target search block not found in '{path}'. "
                f"Ensure exact indentation and matching context."
            )
        if count > 1:
            return (
                f"Error: Target search block appears {count} times in '{path}'. "
                f"Please provide more surrounding lines to create a unique block."
            )

        # Exact unique replacement
        new_content = content.replace(old_str, new_str, 1)
        resolved.write_text(new_content, encoding="utf-8")

        return f"Successfully updated '{path}' (1 replacement applied)."
    except Exception as e:
        return f"Error applying edit to '{path}': {e}"
