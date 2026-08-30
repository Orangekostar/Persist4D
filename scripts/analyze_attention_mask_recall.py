#!/usr/bin/env python3
"""Run manifest-bound mask-attention recall diagnostics."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rescene_rootcause_diagnostic_cli import main_for_mode

if __name__ == "__main__":
    raise SystemExit(main_for_mode("attention_mask_recall"))
