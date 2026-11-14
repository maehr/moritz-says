"""Tests for ``scripts.export_site``."""

import subprocess
from pathlib import Path

import pytest

from scripts import export_site


def test_main_exports_then_prepares_the_html(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> None:
        calls.append(command)
        (tmp_path / "dist").mkdir()
        (tmp_path / "dist" / "index.html").write_text(
            '<meta name="description" content="a marimo app" />', encoding="utf-8"
        )

    monkeypatch.setattr(export_site, "ROOT", tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_run)
    export_site.main()
    assert calls == [export_site.EXPORT_COMMAND]
    assert (tmp_path / "dist" / "index.html").read_text(encoding="utf-8") == ""
