"""Rebuild this article's white PNG figures and SVG exports."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    runpy.run_path(str(root / "source" / "render_charts.py"), run_name="__main__")
