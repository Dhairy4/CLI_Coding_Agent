import sys
from pathlib import Path

# Ensure src/ is in sys.path so coding_agent is always importable without conflict
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from coding_agent.cli import run_cli

if __name__ == "__main__":
    run_cli()
