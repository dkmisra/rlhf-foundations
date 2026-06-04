import sys
from pathlib import Path

# Mirror scripts/run_rl.py: make the package modules (llm, rlhf, tasks, utils)
# importable as top-level packages during tests.
ROOT = Path(__file__).resolve().parents[1]
RLHF2 = ROOT / "rlhf2"
if str(RLHF2) not in sys.path:
    sys.path.insert(0, str(RLHF2))
