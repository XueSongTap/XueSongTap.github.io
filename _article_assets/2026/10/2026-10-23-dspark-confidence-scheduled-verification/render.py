#!/usr/bin/env python3
"""Rebuild all figures relative to this file; editable figure code lives in source/."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parent
local_packages = root / 'build/python_packages'
if local_packages.is_dir():
    sys.path.insert(0, str(local_packages))
runpy.run_path(str(root / 'source/dspark_figures.py'), run_name='__main__')
