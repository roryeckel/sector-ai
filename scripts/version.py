#!/usr/bin/env python3
"""Read, set and check the project version.

``pyproject.toml`` holds the only version literal. The release workflows call
this script before and after the bump so a tag that disagrees with the package
fails the run instead of publishing something mislabelled.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYPROJECT = ROOT / "pyproject.toml"

SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?(\+[0-9A-Za-z.-]+)?$")
VERSION_LINE = re.compile(r'^version = "(?P<version>[^"]+)"$', re.MULTILINE)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT, help="Path to pyproject.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("get", help="Print the current version")

    set_parser = subparsers.add_parser("set", help="Write a new version")
    set_parser.add_argument("version")

    check_parser = subparsers.add_parser("check", help="Fail unless the version matches")
    check_parser.add_argument("version")

    forward_parser = subparsers.add_parser(
        "forward-check", help="Fail unless the new version sorts above the current one"
    )
    forward_parser.add_argument("version")

    canonical_parser = subparsers.add_parser("canonical", help="Print the PEP 440 form used by installed metadata")
    canonical_parser.add_argument("version")

    return parser.parse_args(argv)


def normalize(value: str) -> str:
    return value[1:] if value.startswith("v") else value


def validate(value: str) -> str:
    version = normalize(value)
    if not SEMVER.match(version):
        raise SystemExit(f"not a semantic version: {value!r}")
    return version


def read_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = VERSION_LINE.search(text)
    if not match:
        raise SystemExit(f"could not find the project version in {path}")
    return match.group("version")


def write_version(path: Path, version: str) -> None:
    raw = path.read_bytes().decode("utf-8")
    # Keep the file's existing line endings: CI checks out a CRLF blob on
    # Linux, and writing universal newlines there would rewrite every line.
    newline = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    updated, replacements = VERSION_LINE.subn(f'version = "{version}"', text, count=1)
    if replacements != 1:
        raise SystemExit(f"expected exactly one project version in {path}, found {replacements}")
    path.write_bytes(updated.replace("\n", newline).encode("utf-8"))


def release_key(version: str) -> tuple:
    """Precedence key matching semantic versioning, prereleases included.

    ``1.4.0-alpha`` < ``1.4.0-alpha.1`` < ``1.4.0-beta`` < ``1.4.0-rc.1`` <
    ``1.4.0``. Numeric identifiers compare numerically and sort below
    alphanumeric ones, as the spec requires.
    """
    core, _, pre = normalize(version).partition("+")[0].partition("-")
    core_key = tuple(int(part) for part in core.split("."))
    if not pre:
        return (*core_key, 1, ())
    identifiers = []
    for part in pre.split("."):
        identifiers.append((0, int(part), "") if part.isdigit() else (1, 0, part))
    return (*core_key, 0, tuple(identifiers))


PRE_TAGS = {"a": "a", "alpha": "a", "b": "b", "beta": "b", "c": "rc", "rc": "rc", "pre": "rc", "preview": "rc"}


def canonical(version: str) -> str:
    """Return the PEP 440 form setuptools and importlib.metadata report.

    ``1.4.0-rc.1`` is a valid tag but the installed distribution calls itself
    ``1.4.0rc1``, so comparing ``sector-ai --version`` against the tag needs
    this translation.
    """
    core, _, pre = normalize(version).partition("-")
    if not pre:
        return core
    match = re.fullmatch(r"([A-Za-z]+)\.?(\d*)", pre)
    if not match:
        return normalize(version).lower()
    tag = match.group(1).lower()
    number = match.group(2)
    if tag == "dev":
        return f"{core}.dev{number}" if number else f"{core}.dev"
    return f"{core}{PRE_TAGS.get(tag, tag)}{number}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.pyproject

    if args.command == "get":
        print(read_version(path))
        return 0

    version = validate(args.version)

    if args.command == "canonical":
        print(canonical(version))
        return 0

    if args.command == "set":
        write_version(path, version)
        print(f"version set to {version}")
        return 0

    current = read_version(path)

    if args.command == "check":
        if canonical(current) != canonical(version):
            raise SystemExit(f"pyproject.toml has {current}, expected {version}")
        print(f"version check passed ({version})")
        return 0

    # forward-check
    if release_key(version) <= release_key(current):
        raise SystemExit(f"{version} does not sort above the current {current}")
    print(f"{current} -> {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
