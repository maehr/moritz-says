"""Guard the notebook layout. CI does not run the browser tests."""

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


def cell_sources() -> list[str]:
    source = APP.read_text(encoding="utf-8")
    cells: list[str] = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if any(ast.unparse(decorator) == "app.cell" for decorator in node.decorator_list):
            cells.append(
                "\n".join(ast.get_source_segment(source, stmt) or "" for stmt in node.body)
            )
    return cells


def test_no_two_cells_have_the_same_code() -> None:
    """A merge once left the old cells beside the new ones. The page then showed each twice."""
    cells = cell_sources()
    assert len(cells) > 5
    duplicates = {cell for cell in cells if cells.count(cell) > 1}
    assert not duplicates


def test_page_parts_appear_once() -> None:
    source = APP.read_text(encoding="utf-8")
    for part in (
        'class="site-header"',
        'class="site-footer"',
        '<h2 id="about-heading">',
        "mo.routes(",
    ):
        assert source.count(part) == 1, part
