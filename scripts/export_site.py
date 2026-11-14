"""Export the site. Drop marimo's default description, so that the page has one.

Run ``uv run python -m scripts.export_site`` from the repository root.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_COMMAND = [
    sys.executable,
    "-m",
    "marimo",
    "export",
    "html-wasm",
    "app.py",
    "-o",
    "dist",
    "--mode",
    "run",
]


def prepare_html(raw: str) -> str:
    """Remove the default marimo description. ``head.html`` supplies the real one.

    >>> prepare_html('<meta name="description" content="a marimo app" /><meta name="description" content="Mine">')
    '<meta name="description" content="Mine">'
    >>> prepare_html('<meta name="description" content="Mine">')
    '<meta name="description" content="Mine">'
    """
    return re.sub(r'<meta name="description" content="a marimo app"\s*/?>', "", raw)


def main() -> None:
    """Export the notebook, then prepare ``dist/index.html``.

    >>> callable(main)
    True
    """
    subprocess.run(EXPORT_COMMAND, cwd=ROOT, check=True)
    path = ROOT / "dist" / "index.html"
    path.write_text(prepare_html(path.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()
