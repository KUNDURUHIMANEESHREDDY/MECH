"""Export all code from every file in the MECH repository into a single text file.

Generates a unified text file containing every source code file in MECH
(backend, tests, frontend, scripts, automation, CI and project configs),
prefaced with an exact index indicating line numbers and line counts for
each file.

Usage:
    python scripts/export_all_source.py [--output PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_DIRS = {
    ".git", ".claude", ".agents", ".agent-factory", "__pycache__",
    "node_modules", ".pytest-tmp", "dist", "build", "release",
    ".venv", "venv", ".cache", "coverage", ".idea", ".vscode",
    ".pytest_cache", ".omo", "screenshots", "test-results", "design-demos"
}

EXCLUDE_FILES = {
    "MECH_all_source.txt", "MECH_all_source_old.txt", "_full.txt", "_ma.txt",
    "package-lock.json"
}

EXCLUDE_EXTS = {
    ".png", ".ico", ".jpg", ".jpeg", ".gif", ".svg", ".exe", ".dll",
    ".pak", ".bin", ".zip", ".asar", ".db", ".sqlite", ".sqlite3",
    ".pyc", ".pyo", ".pyd", ".blockmap", ".dat", ".err", ".patch"
}


def collect_files() -> List[str]:
    """Collect all genuine source files across the project."""
    files_to_include = []

    # 1. Root code files
    for f in os.listdir(ROOT):
        p = ROOT / f
        if p.is_file() and f not in EXCLUDE_FILES and not f.startswith("."):
            if f.endswith((".py", ".js", ".ini", ".txt")):
                files_to_include.append(f)

    # 2. CI workflow
    ci_path = ROOT / ".github" / "workflows" / "ci.yml"
    if ci_path.exists():
        files_to_include.append(".github/workflows/ci.yml")

    # 3. backend
    for root, dirs, files in os.walk(ROOT / "backend"):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".py"):
                rel = Path(root, f).relative_to(ROOT).as_posix()
                files_to_include.append(rel)

    # 4. tests
    for root, dirs, files in os.walk(ROOT / "tests"):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith(".py"):
                rel = Path(root, f).relative_to(ROOT).as_posix()
                files_to_include.append(rel)

    # 5. scripts
    for root, dirs, files in os.walk(ROOT / "scripts"):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f.endswith((".py", ".js", ".cjs")):
                rel = Path(root, f).relative_to(ROOT).as_posix()
                files_to_include.append(rel)

    # 6. frontend
    for sub in ["frontend/src", "frontend/electron", "frontend/tests", "frontend/scripts"]:
        sub_dir = ROOT / sub
        if not sub_dir.exists():
            continue
        for root, dirs, files in os.walk(sub_dir):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                if f.endswith((".vue", ".ts", ".js", ".css", ".html")):
                    rel = Path(root, f).relative_to(ROOT).as_posix()
                    files_to_include.append(rel)

    frontend_configs = [
        "frontend/package.json",
        "frontend/vite.config.mts",
        "frontend/vitest.config.js",
        "frontend/playwright.config.js",
        "frontend/tsconfig.json",
        "frontend/index.html",
    ]
    for cfg in frontend_configs:
        p = ROOT / cfg
        if p.exists():
            files_to_include.append(cfg)

    # Sort deterministically
    return sorted(set(files_to_include))


def export_all(output_path: Path) -> Tuple[int, int]:
    """Assemble and write all source files into a single text file.

    Returns (file_count, total_lines).
    """
    file_list = collect_files()

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
        out.write("  build output, test artifacts, binaries, and scratch scripts.\n\n\n")
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

    return len(loaded_files), total_code_lines


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

    print(f"Collecting files and building unified source export...")
    file_count, line_count = export_all(out_path)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done! Wrote {file_count} files ({line_count} lines of code) to {out_path.name}")
    print(f"File size: {size_mb:.2f} MB ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
