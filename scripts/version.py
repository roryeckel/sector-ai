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


def release_key(version: str) -> tuple[int, ...]:
    """Sort key for the numeric part; a suffix counts as the same release."""
    core = re.split(r"[-+]", version, maxsplit=1)[0]
    return tuple(int(part) for part in core.split("."))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.pyproject

    if args.command == "get":
        print(read_version(path))
        return 0

    version = validate(args.version)

    if args.command == "set":
        write_version(path, version)
        print(f"version set to {version}")
        return 0

    current = read_version(path)

    if args.command == "check":
        if current != version:
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
