"""Export all code from every file in the MECH repository into a single text file.

Generates a unified text file containing every source code file in MECH
(backend, tests, frontend, scripts, automation, CI and project configs),
prefaced with an exact index indicating line numbers and line counts for
each file.

Every candidate passes `scripts.export_guard` before it is embedded: known
credential paths are dropped, high-confidence secret contents are dropped, and
unrecognised file types are dropped rather than published. The declined files
and the reason for each are written to `export_all_source_manifest.json` beside
the output, and summarised in the output header -- so the "Complete Source
Code Repository" title now comes with the counts of what was withheld.

Usage:
    python scripts/export_all_source.py [--output PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_guard import (  # noqa: E402
    ScreenReport,
    format_summary,
    screen_file,
    write_manifest,
)

EXCLUDE_DIRS = {
    ".git", ".claude", ".agents", ".agent-factory", "__pycache__",
    "node_modules", ".pytest-tmp", "dist", "build", "release",
    ".venv", "venv", ".cache", "coverage", ".idea", ".vscode",
    ".pytest_cache", ".omo", "screenshots", "test-results", "design-demos",
    "exports",
}

#: Known prior-export names. The *resolved* output path is excluded separately
#: and unconditionally, because `--output snapshot.txt` is not in this set and
#: excluding the default name is not a general defence.
EXCLUDE_FILES = {
    "MECH_all_source.txt", "MECH_all_source_old.txt", "_full.txt", "_ma.txt",
    "MECH_backend.txt", "MECH_frontend.txt", "MECH_tests.txt",
    "MECH_scripts.txt", "MECH_root.txt",
    "package-lock.json"
}

#: Areas walked whole, every recognised text file included. Previously each area
#: carried its own inline suffix allow-list and the root scan accepted only
#: `.py/.js/.ini/.txt`, so `package.json`, `pyproject.toml`, `Makefile` and
#: `.pre-commit-config.yaml` were silently missing from a file titled
#: "Complete Source Code Repository". Coverage is now a property of the walk
#: and the guard alone decides publishability.
SOURCE_AREAS = ("backend", "tests", "scripts", "frontend/src",
                "frontend/electron", "frontend/tests", "frontend/scripts")


def _escapes_repo(path: Path) -> bool:
    """Whether `path` is a symlink resolving outside the repository."""
    try:
        return not path.resolve().is_relative_to(ROOT.resolve())
    except (OSError, ValueError):
        return True


def collect_files(report: ScreenReport,
                  excluded_paths: Optional[Set[Path]] = None) -> List[str]:
    """Collect every publishable source file, recording what was declined.

    Returns the sorted relative paths to embed. Every decline lands on `report`,
    so the caller's counts account for the whole tree rather than for the part
    of it that survived.
    """
    files_to_include: List[str] = []
    seen: Set[str] = set()
    skip = excluded_paths or set()

    def consider(path: Path, rel: str) -> None:
        if rel in seen:
            return
        seen.add(rel)
        if path in skip:
            report.reject(rel, "exporter-output")
            return
        if rel in EXCLUDE_FILES or rel.endswith(".log.err"):
            report.reject(rel, "excluded-by-name")
            return
        if path.is_symlink() and _escapes_repo(path):
            # A link to $HOME/.ssh/id_rsa or to a path outside the tree would
            # otherwise be read and embedded verbatim.
            report.reject(rel, "symlink-escapes-repo")
            return
        if screen_file(path, rel, report):
            report.accept(rel)
            files_to_include.append(rel)

    # 1. Root-level files and dotfiles, project config included.
    for entry in sorted(ROOT.iterdir()):
        if entry.is_file():
            consider(entry, entry.name)

    # 2. CI workflows, all of them. The previous scan hardcoded `ci.yml`, so
    #    every other workflow was absent.
    workflows = ROOT / ".github" / "workflows"
    if workflows.is_dir():
        for entry in sorted(workflows.rglob("*")):
            if entry.is_file():
                consider(entry, entry.relative_to(ROOT).as_posix())

    # 3. Source areas, and frontend's own root-level configs.
    for area in SOURCE_AREAS:
        base = ROOT / area
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
            for name in sorted(filenames):
                path = Path(dirpath) / name
                consider(path, path.relative_to(ROOT).as_posix())

    frontend = ROOT / "frontend"
    if frontend.is_dir():
        for entry in sorted(frontend.iterdir()):
            if entry.is_file():
                consider(entry, entry.relative_to(ROOT).as_posix())

    return sorted(set(files_to_include))


def export_all(output_path: Path) -> Tuple[int, int, ScreenReport]:
    """Assemble and write all source files into a single text file.

    Returns (file_count, total_lines, report).
    """
    report = ScreenReport()
    # The output path is excluded whether or not it matches EXCLUDE_FILES: a
    # custom `--output` inside the repository would otherwise be discovered as
    # a root `.txt` file and embedded in the snapshot it just wrote.
    excluded_paths = {output_path.resolve()}
    file_list = collect_files(report, excluded_paths)

    # Pass 1: Read all contents and normalize newlines
    loaded_files: List[Tuple[str, List[str]]] = []
    total_code_lines = 0
    package_inits = 0

    for rel_path in file_list:
        full_path = ROOT / rel_path
        with open(full_path, "r", encoding="utf-8", errors="replace") as fh:
            raw_text = fh.read()
        lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        # If the file ended with a newline (standard), pop the trailing empty string
        # so line count matches line count of file
        if lines and lines[-1] == "":
            lines.pop()
        loaded_files.append((rel_path, lines))
        total_code_lines += len(lines)
        if rel_path.endswith("__init__.py"):
            package_inits += 1

    # Pass 2: Calculate line positions
    # We will format the header and index, and calculate the exact line offset
    # for each file.
    #
    # Header format:
    # Line 1: ==============================================================================
    # Line 2: MECH - Complete Source Code Repository
    # Line 3: ==============================================================================
    # Line 4:
    # Line 5:   {count} files, {total_code_lines} lines of code.
    # Line 6:   Includes all {package_inits} package __init__.py files.
    # Line 7:   Contains backend, tests, frontend, scripts, automation, CI and configs.
    # Line 8:   Excluded: __pycache__, node_modules, .git, .claude worktrees,
    # Line 9:   build output, test artifacts, binaries, and scratch scripts.
    # Line 10:
    # Line 11:
    # Line 12: INDEX  ({count} files)
    # Line 13: ------------------------------------------------------------------------------
    # Line 14:   Line (content)  Lines  Path
    # Line 15: ------------------------------------------------------------------------------
    # Line 16 .. 16 + count - 1: Index rows
    # Line 16 + count:
    # Line 17 + count:
    # Then each file:
    #   ==============================================================================
    #   ### {rel_path}   ({count} lines)
    #   ==============================================================================
    #   line 1 .. line N
    #   <blank line>

    header_lines_before_index = 15
    index_entry_count = len(loaded_files)
    header_lines_after_index = 2

    # First file's banner starts at:
    # 1-indexed: 15 + index_entry_count + 2 + 1
    current_line = header_lines_before_index + index_entry_count + header_lines_after_index + 1

    index_rows = []
    file_offsets = []

    for rel_path, lines in loaded_files:
        line_count = len(lines)
        content_start = current_line + 3  # Banner takes 3 lines, content starts on line 4 of block
        file_offsets.append((current_line, content_start, line_count))
        index_rows.append(f"{content_start:>10}     {line_count:>5}  {rel_path}")
        # Banner (3 lines) + content (line_count lines) + 1 blank line = line_count + 4
        current_line += line_count + 4

    # Build full document in memory / stream to file
    with open(output_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("==============================================================================\n")
        out.write("MECH - Complete Source Code Repository\n")
        out.write("==============================================================================\n\n")
        out.write(f"  {len(loaded_files)} files, {total_code_lines} lines of code.\n")
        out.write(f"  Includes all {package_inits} package __init__.py files.\n")
        out.write("  Contains backend, tests, frontend, scripts, automation, CI and configs.\n")
        out.write("  Excluded: __pycache__, node_modules, .git, .claude worktrees,\n")
        out.write("  build output, test artifacts, binaries, and scratch scripts.\n")
        out.write(f"  Secret / unknown-type screening: {format_summary(report)}.\n\n\n")
        out.write(f"INDEX  ({len(loaded_files)} files)\n")
        out.write("------------------------------------------------------------------------------\n")
        out.write("   Content   Lines  Path\n")
        out.write("------------------------------------------------------------------------------\n")
        for row in index_rows:
            out.write(row + "\n")
        out.write("\n\n")

        for (rel_path, lines), (banner_start, content_start, line_count) in zip(loaded_files, file_offsets):
            out.write("==============================================================================\n")
            out.write(f"### {rel_path}   ({line_count} lines)\n")
            out.write("==============================================================================\n")
            for line in lines:
                out.write(line + "\n")
            out.write("\n")

    # Named after the output it describes, so `--output snapshot.txt` does not
    # leave a manifest called `export_all_source_*` next to it, and so the
    # ignore rule can follow the output rather than the script.
    stem = output_path.stem or "export_all_source"
    write_manifest(output_path.parent, report, prefix=stem)
    return len(loaded_files), total_code_lines, report


def main():
    parser = argparse.ArgumentParser(description="Export all MECH source code into one text file")
    parser.add_argument(
        "--output", "-o",
        default="MECH_all_source.txt",
        help="Target output text file (default: MECH_all_source.txt in repo root)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    print("Collecting files and building unified source export...")
    file_count, line_count, report = export_all(out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done! Wrote {file_count} files ({line_count} lines of code) to {out_path.name}")
    print(f"File size: {size_mb:.2f} MB ({out_path.stat().st_size:,} bytes)")
    print(f"GUARD: {format_summary(report)}")


if __name__ == "__main__":
    main()
