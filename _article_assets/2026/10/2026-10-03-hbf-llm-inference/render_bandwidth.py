#!/usr/bin/env python3
"""Rebuild bandwidth-budget assets with Python + matplotlib."""
from pathlib import Path
import runpy
import sys
ASSET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ASSET_ROOT.parents[3]
local_dependencies = ASSET_ROOT / '.render-deps'
if local_dependencies.exists():
    sys.path.insert(0, str(local_dependencies))
runpy.run_path(str(ASSET_ROOT/'source'/'render_bandwidth_budget.py'),
              init_globals={'ASSET_ROOT': ASSET_ROOT, 'REPO_ROOT': REPO_ROOT})
