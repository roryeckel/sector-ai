"""Tests for scripts/version.py, which all three release workflows depend on."""

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "version.py"
CURRENT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def run(pyproject: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--pyproject", str(pyproject), *args],
        capture_output=True,
        text=True,
    )


def write_pyproject(tmp_path: Path, version: str) -> Path:
    target = tmp_path / "pyproject.toml"
    target.write_text(f'[project]\nname = "sector-ai"\nversion = "{version}"\n', encoding="utf-8")
    return target


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


@pytest.mark.parametrize(
    ("current", "candidate"),
    [
        ("1.4.0", "1.4.0"),
        ("1.4.1", "1.4.0"),
        ("2.0.0", "1.9.9"),
        ("1.4.0-rc.1", "1.4.0-rc.1"),
    ],
)
def test_forward_check_rejects_the_same_or_lower_version(tmp_path, current, candidate):
    """Explicit fixtures: deriving a lower version from the repo's own spelling
    breaks while the repo sits on a prerelease."""
    assert run(write_pyproject(tmp_path, current), "forward-check", candidate).returncode == 1


def test_forward_check_accepts_a_prerelease_of_a_higher_version(pyproject):
    major = int(CURRENT.split(".")[0])
    assert run(pyproject, "forward-check", f"{major + 1}.0.0-rc.1").returncode == 0


@pytest.mark.parametrize(
    ("current", "candidate"),
    [
        ("1.4.0-alpha", "1.4.0-alpha.1"),
        ("1.4.0-alpha.1", "1.4.0-beta"),
        ("1.4.0-beta", "1.4.0-rc.1"),
        ("1.4.0-rc.1", "1.4.0-rc.2"),
        ("1.4.0-rc.2", "1.4.0"),
        ("1.4.0", "1.4.1"),
    ],
)
def test_forward_check_orders_prereleases(tmp_path, current, candidate):
    """rc.1 -> rc.2 -> final must all sort forward, or prereleases are unusable."""
    assert run(write_pyproject(tmp_path, current), "forward-check", candidate).returncode == 0


@pytest.mark.parametrize(
    ("current", "candidate"),
    [
        ("1.4.0", "1.4.0-rc.1"),
        ("1.4.0-rc.2", "1.4.0-rc.1"),
        ("1.4.0-beta", "1.4.0-alpha"),
        ("1.4.1", "1.4.0"),
    ],
)
def test_forward_check_rejects_a_lower_version(tmp_path, current, candidate):
    assert run(write_pyproject(tmp_path, current), "forward-check", candidate).returncode == 1


@pytest.mark.parametrize(
    "version",
    [
        "1.3.0",
        "1.4.0-rc.1",
        "1.4.0-rc1",
        "1.4.0-alpha.2",
        "1.4.0-beta.10",
        "1.5.0-dev.3",
        # No trailing number: PEP 440 pads these with 0 (1.4.0a0).
        "1.4.0-alpha",
        "1.4.0-beta",
        "1.4.0-rc",
        "1.4.0-preview",
        "1.4.0-dev",
    ],
)
def test_canonical_matches_packaging(version):
    result = run(ROOT / "pyproject.toml", "canonical", version)

    assert result.returncode == 0
    assert result.stdout.strip() == str(Version(version))


def test_check_accepts_equivalent_spellings(tmp_path):
    """The tag may say rc.1 while installed metadata says rc1."""
    target = write_pyproject(tmp_path, "1.4.0rc1")

    assert run(target, "check", "1.4.0-rc.1").returncode == 0


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
