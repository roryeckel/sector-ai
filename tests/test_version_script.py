"""Tests for scripts/version.py, which all three release workflows depend on."""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "version.py"
CURRENT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def run(pyproject: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--pyproject", str(pyproject), *args],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def pyproject(tmp_path):
    target = tmp_path / "pyproject.toml"
    shutil.copy(ROOT / "pyproject.toml", target)
    return target


def test_get_reports_the_pyproject_version(pyproject):
    result = run(pyproject, "get")

    assert result.returncode == 0
    assert result.stdout.strip() == CURRENT


def test_check_accepts_the_current_version(pyproject):
    assert run(pyproject, "check", CURRENT).returncode == 0
    assert run(pyproject, "check", f"v{CURRENT}").returncode == 0


def test_check_rejects_a_different_version(pyproject):
    result = run(pyproject, "check", "999.0.0")

    assert result.returncode == 1
    assert "999.0.0" in result.stderr


def test_set_changes_only_the_version_line(pyproject):
    before = pyproject.read_text(encoding="utf-8").splitlines()

    assert run(pyproject, "set", "9.9.9").returncode == 0
    after = pyproject.read_text(encoding="utf-8").splitlines()

    assert run(pyproject, "get").stdout.strip() == "9.9.9"
    assert len(before) == len(after)
    assert [i for i, (a, b) in enumerate(zip(before, after)) if a != b] == [
        next(i for i, line in enumerate(before) if line.startswith("version = "))
    ]


def test_set_preserves_line_endings(tmp_path):
    source = ROOT / "pyproject.toml"
    target = tmp_path / "pyproject.toml"
    target.write_bytes(source.read_bytes())
    before = source.read_bytes().count(b"\r\n")

    assert run(target, "set", "9.9.9").returncode == 0

    assert target.read_bytes().count(b"\r\n") == before


def test_forward_check_accepts_a_higher_version(pyproject):
    assert run(pyproject, "forward-check", "999.0.0").returncode == 0


def test_forward_check_rejects_the_same_or_lower_version(pyproject):
    assert run(pyproject, "forward-check", CURRENT).returncode == 1

    lower = CURRENT.rsplit(".", 1)[0] + ".0"
    if lower != CURRENT:
        assert run(pyproject, "forward-check", lower).returncode == 1


def test_forward_check_accepts_a_prerelease_of_a_higher_version(pyproject):
    major = int(CURRENT.split(".")[0])
    assert run(pyproject, "forward-check", f"{major + 1}.0.0-rc.1").returncode == 0


def test_rejects_a_non_semver(pyproject):
    result = run(pyproject, "forward-check", "banana")

    assert result.returncode == 1
    assert "not a semantic version" in result.stderr


def test_missing_version_line_is_reported(tmp_path):
    target = tmp_path / "pyproject.toml"
    target.write_text('[project]\nname = "sector-ai"\n', encoding="utf-8")

    result = run(target, "get")

    assert result.returncode == 1
    assert "could not find" in result.stderr
