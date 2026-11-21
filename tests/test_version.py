"""Tests for the package version and its fallback."""

import importlib
import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import sector_ai


def test_installed_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    expected = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]

    assert sector_ai.__version__ == expected


def test_version_falls_back_when_not_installed(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", missing)
    try:
        reloaded = importlib.reload(sector_ai)
        assert reloaded.__version__ == "unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(sector_ai)
