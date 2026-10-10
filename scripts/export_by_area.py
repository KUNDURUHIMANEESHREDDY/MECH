"""Export all code grouped by AREA (backend, frontend, tests, ...) into .txt files.

Everything this publishes passes `scripts.export_guard` first: a known credential
path is dropped, a file whose contents match a high-confidence secret pattern is
dropped, and an unrecognised file type is dropped rather than bucketed as
`other`. The dropped files and the reason for each are written to
`_manifest.json` beside the exports.

Usage:
    python scripts/export_by_area.py
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_guard import (  # noqa: E402
    ScreenReport,
    format_summary,
    screen_file,
    write_manifest,
)

OUT = ROOT / "exports" / "code_by_area"
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE_DIRS = {".git", ".agents", "__pycache__", "node_modules", "dist", "build",
                "release", ".venv", "venv", ".cache", "coverage", ".pytest_cache",
                "screenshots", "test-results", "design-demos", ".idea", ".vscode",
                "exports", "win-unpacked", ".pytest-tmp"}

EXCLUDE_FILES = {"package-lock.json"}


def area_of(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) == 1:
        return "root"
    top = parts[0]
    if top in ("backend", "frontend", "tests", "scripts", "docs"):
        return top
    if top == ".github":
        return "github"
    return "other"


def collect():
    """Group every publishable file by area, recording what was declined.

    The screening decision lives in `export_guard` so all three exporters apply
    one policy. This function only decides *which bucket* a survivor goes in.
    """
    report = ScreenReport()
    grouped: dict = {}
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn
                 if d not in EXCLUDE_DIRS and not d.startswith(".pytest-tmp")]
        for f in fn:
            p = Path(dp) / f
            rel = p.relative_to(ROOT).as_posix()
            if f in EXCLUDE_FILES:
                report.reject(rel, "excluded-by-name")
                continue
            if f.endswith(".log.err"):
                report.reject(rel, "excluded-by-name")
                continue
            if not screen_file(p, rel, report):
                continue
            report.accept(rel)
            a = area_of(rel)
            grouped.setdefault(a, []).append(rel)
    for a in grouped:
        grouped[a].sort()
    return grouped, report


def export():
    grouped, report = collect()
    total = sum(len(v) for v in grouped.values())
    # Recreated per run rather than only at import: the directory is created
    # once at module load, so a deleted output directory failed on first write.
    OUT.mkdir(parents=True, exist_ok=True)
    for a, files in sorted(grouped.items()):
        out_path = OUT / f"{a}.txt"
        total_lines = 0
        with open(out_path, "w", encoding="utf-8", newline="\n") as out:
            out.write(f"===== {a.upper()} — {len(files)} files =====\n\nINDEX\n-----\n")
            for rel in files:
                out.write(f"{rel}\n")
            out.write("\n\n")
            for rel in files:
                fp = ROOT / rel
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    text = f"<unreadable: {e}>"
                lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                if lines and lines[-1] == "":
                    lines.pop()
                total_lines += len(lines)
                out.write("=" * 78 + "\n")
                out.write(f"### {rel} ({len(lines)} lines)\n")
                out.write("=" * 78 + "\n")
                for ln in lines:
                    out.write(ln + "\n")
                out.write("\n")
        size_mb = out_path.stat().st_size / 1048576
        print(f"{a}.txt: {len(files)} files, {total_lines} lines, {size_mb:.2f} MB")
    manifest = write_manifest(OUT, report)
    print(f"TOTAL: {total} files -> {OUT}")
    print(f"GUARD: {format_summary(report)}")
    if manifest is not None:
        print(f"MANIFEST: {manifest}")
    return report


if __name__ == "__main__":
    export()
