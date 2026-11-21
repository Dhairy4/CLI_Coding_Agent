from pathlib import Path
import pytest
from coding_agent.tools.filesystem import apply_search_replace, read_file_lines, resolve_safe_path
from coding_agent.prompts.builder import assemble_prompt_messages, STATIC_SYSTEM_PREFIX
from coding_agent.state import WorkingMemory
from coding_agent.context.snapshot import collect_repo_snapshot
from coding_agent.context.indexer import ASTCodeIndexer


def test_path_jailing(tmp_path: Path):
    """Verify that path resolution forbids escaping workspace root."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Valid relative path inside root
    valid_path = resolve_safe_path("src/code.py", workspace_root=workspace)
    assert valid_path == workspace / "src" / "code.py"

    # Traversal attack
    with pytest.raises(PermissionError):
        resolve_safe_path("../secret.txt", workspace_root=workspace)

    with pytest.raises(PermissionError):
        resolve_safe_path("/etc/passwd", workspace_root=workspace)


def test_apply_search_replace_guarantees(tmp_path: Path):
    """Verify apply_search_replace refuses 0 or >1 matches and succeeds on 1 exact match."""
    workspace = tmp_path
    sample_file = workspace / "sample.py"
    sample_file.write_text("def foo():\n    return 1\n\ndef bar():\n    return 1\n", encoding="utf-8")

    from coding_agent.config import config
    orig_root = config.workspace_root
    config.workspace_root = workspace

    try:
        # 1. Zero matches
        res_zero = apply_search_replace.invoke({"path": "sample.py", "old_str": "def baz():", "new_str": "def baz(): pass"})
        assert "not found" in res_zero

        # 2. Ambiguous (>1 matches)
        res_multi = apply_search_replace.invoke({"path": "sample.py", "old_str": "    return 1", "new_str": "    return 2"})
        assert "appears 2 times" in res_multi

        # 3. Exactly 1 match
        target_block = "def foo():\n    return 1"
        replacement_block = "def foo():\n    return 42"
        res_success = apply_search_replace.invoke({"path": "sample.py", "old_str": target_block, "new_str": replacement_block})
        assert "Successfully updated" in res_success
        assert "return 42" in sample_file.read_text(encoding="utf-8")
        assert "return 1" in sample_file.read_text(encoding="utf-8")  # bar still has return 1

    finally:
        config.workspace_root = orig_root


def test_read_file_lines(tmp_path: Path):
    workspace = tmp_path
    sample_file = workspace / "test.txt"
    sample_file.write_text("\n".join(f"line_{i}" for i in range(1, 21)), encoding="utf-8")

    from coding_agent.config import config
    orig_root = config.workspace_root
    config.workspace_root = workspace

    try:
        content = read_file_lines.invoke({"path": "test.txt", "start_line": 5, "line_count": 3})
        assert "line_5" in content
        assert "line_6" in content
        assert "line_7" in content
        assert "line_8" not in content
    finally:
        config.workspace_root = orig_root


def test_prompt_cache_boundaries():
    """Verify static prefix and dynamic working memory positioning."""
    memory = WorkingMemory(
        current_goal="Fix bug in parser",
        plan_steps=["Read AST", "Apply patch"],
        files_modified=["parser.py"],
    )
    state = {
        "messages": [],
        "workspace_root": "/tmp",
        "repo_snapshot": "Branch: main",
        "approval_mode": "ask",
        "working_memory": memory,
        "step_count": 0,
        "max_steps": 25,
        "pending_approval": None,
        "last_command_output": None,
    }

    assembled = assemble_prompt_messages(state)
    assert len(assembled) == 2  # static system prefix + working memory system msg
    assert STATIC_SYSTEM_PREFIX in assembled[0].content
    assert "Branch: main" in assembled[0].content
    assert "Fix bug in parser" in assembled[1].content
    assert "parser.py" in assembled[1].content


def test_ast_symbol_extraction(tmp_path: Path):
    """Verify Tree-sitter extracts functions and classes accurately."""
    py_file = tmp_path / "calc.py"
    py_file.write_text(
        "import math\n\nclass Calculator:\n    def add(self, a, b):\n        return a + b\n\ndef multiply(x, y):\n    return x * y\n"
    )

    indexer = ASTCodeIndexer(workspace_root=tmp_path)
    symbols = indexer.extract_symbols(py_file)

    symbol_names = [s.symbol_name for s in symbols]
    assert "import math" in symbol_names
    assert "Calculator" in symbol_names
    assert "Calculator.add" in symbol_names
    assert "multiply" in symbol_names
