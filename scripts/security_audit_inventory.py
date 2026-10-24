#!/usr/bin/env python3
"""Build an exact installed-package inventory for pip-audit.

pip-audit's PyPI service cannot resolve PyTorch's local +cpu build label.
For that one known wheel, audit the corresponding public upstream version.
All other local build labels fail closed until their mapping is reviewed.
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib.metadata import distributions
from pathlib import Path
from typing import Iterable


def canonical_name(name: str) -> str:
    """Normalize a distribution name according to Python package-name rules."""
    return re.sub(r"[-_.]+", "-", name).lower()


def normalize_package(name: str, version: str) -> tuple[str, str, bool]:
    """Return canonical name, auditable version and whether a build label changed."""
    package = canonical_name(name)
    if not package or not version:
        raise ValueError(f"Missing package name or version: {name!r} {version!r}")

    if "+" not in version:
        return package, version, False

    public_version, local_label = version.rsplit("+", 1)
    if package == "torch" and local_label.casefold() == "cpu" and public_version:
        return package, public_version, True

    raise ValueError(
        f"Unsupported local build label for {name}=={version}. "
        "Refusing to silently exclude or rewrite this installed package."
    )


def render_inventory(entries: Iterable[tuple[str, str]]) -> tuple[str, list[str]]:
    """Render a deterministic, exactly pinned requirements file from installed metadata."""
    packages: dict[str, str] = {}
    normalized: list[str] = []

    for name, installed_version in entries:
        package, audit_version, changed = normalize_package(name, installed_version)
        previous = packages.get(package)
        if previous is not None and previous != audit_version:
            raise ValueError(
                f"Conflicting installed versions for {package}: "
                f"{previous} and {audit_version}"
            )
        packages[package] = audit_version
        if changed:
            normalized.append(
                f"{name}=={installed_version} -> {package}=={audit_version} "
                "(audit upstream release; CPU build suffix only)"
            )

    if not packages:
        raise ValueError("No installed Python distributions were found to audit.")

    content = "".join(f"{name}=={packages[name]}\n" for name in sorted(packages))
    return content, normalized


def installed_entries() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for distribution in distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name and version:
            entries.append((name, version))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mech-audit-requirements.txt"),
        help="destination for the generated, exactly pinned inventory",
    )
    args = parser.parse_args()

    try:
        content, normalized = render_inventory(installed_entries())
    except ValueError as exc:
        print(f"Security inventory generation failed closed: {exc}", file=sys.stderr)
        return 2

    args.output.write_text(content, encoding="utf-8")
    for item in normalized:
        print(f"Reviewed version normalization: {item}")
    print(f"Wrote {len(content.splitlines())} exact installed-package pins to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
