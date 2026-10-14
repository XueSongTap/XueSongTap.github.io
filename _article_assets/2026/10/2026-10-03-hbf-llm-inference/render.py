#!/usr/bin/env python3
"""Rebuild this article asset relative to the staging/repository root."""
from pathlib import Path
import sys
import runpy
ASSET_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ASSET_ROOT.parents[3]
# Asset root lives at REPO_ROOT/_article_assets/YYYY/MM/article-slug.
local_dependencies = ASSET_ROOT / '.render-deps'
if local_dependencies.exists():
    sys.path.insert(0, str(local_dependencies))
runpy.run_path(str(ASSET_ROOT / 'source' / 'render_chart.py'),
               init_globals={'ASSET_ROOT': ASSET_ROOT, 'REPO_ROOT': REPO_ROOT})
