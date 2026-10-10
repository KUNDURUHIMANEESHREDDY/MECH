"""Export all code grouped by language TYPE into .txt files (type, not extension).

Everything this publishes passes `scripts.export_guard` first. In particular the
previous `other` bucket is gone: an unrecognised file is *excluded*, not
exported. Classifying a file nobody has looked at into a bucket that gets
published is how `.env` and `credentials.json` reached a shareable snapshot --
`.json` is an exportable extension, so those files had two routes out.

The dropped files and the reason for each are written to `_manifest.json`.

Usage:
    python scripts/export_by_type.py
"""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_guard import (  # noqa: E402
    BARE_FILENAMES,
    EXCLUDED_EXTENSIONS,
    TEXT_DOTFILES,
    ScreenReport,
    format_summary,
    read_text,
    screen_file,
    write_manifest,
)

OUT = ROOT / "exports" / "code_by_type"
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE_DIRS = {".git", ".agents", "__pycache__", "node_modules", "dist", "build",
                "release", ".venv", "venv", ".cache", "coverage", ".pytest_cache",
                "screenshots", "test-results", "design-demos", ".idea", ".vscode",
                "exports", ".pytest-tmp"}

EXCLUDE_FILES = {"package-lock.json"}

# TYPE mapping (type -> extensions). Type is language, not extension.
TYPE_MAP = {
    "python": {".py"},
    "javascript": {".js", ".cjs", ".mjs"},
    "typescript": {".ts", ".mts", ".cts"},
    "vue": {".vue"},
    "css": {".css"},
    "html": {".html", ".htm"},
    "json": {".json", ".jsonc", ".jsonl"},
    "markdown": {".md", ".mdx", ".rst"},
    "yaml": {".yml", ".yaml"},
    "config": {".ini", ".toml", ".cfg", ".conf", ".properties"},
    "text": {".txt"},
    "shell": {".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd"},
    "sql": {".sql"},
}


def classify(path: Path) -> str:
    """Best-effort language bucket for an already-screened file.

    Only ever called on files `export_guard` accepted, so this cannot be the
    thing that decides to publish an unknown file: `None` means unrecognized
    and the caller excludes it.
    """
    ext = path.suffix.lower()
    name = path.name
    for t, exts in TYPE_MAP.items():
        if ext in exts or name in exts:
            return t
    if ext in EXCLUDED_EXTENSIONS:
        return None
    if name in TEXT_DOTFILES:
        return "config"
    # Extensionless survivors reached here because `export_guard` allow-listed
    # them by name (`BARE_FILENAMES`). Sniff for a language, then fall back to
    # `config` -- `Makefile`, `LICENSE` and friends are build metadata. Anything
    # not on that allow-list never reaches this function.
    if not ext:
        if name in BARE_FILENAMES:
            return "config"
        text = read_text(path)
        if text is not None:
            head = text[:2000]
            first = head.splitlines()[0] if head else ""
            if first.startswith("#!") and "python" in first:
                return "python"
            if first.startswith("#!") and ("/bin/bash" in first or "/bin/sh" in first):
                return "shell"
            if "FROM " in head and "RUN " in head:
                return "docker"
        if name.lower().startswith("dockerfile"):
            return "docker"
        return None
    return None


def collect():
    """Group every publishable file by type, recording what was declined."""
    report = ScreenReport()
    grouped: dict = {}
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn
                 if d not in EXCLUDE_DIRS and not d.startswith(".pytest-tmp")]
        for f in fn:
            p = Path(dp) / f
            rel = p.relative_to(ROOT).as_posix()
            if f in EXCLUDE_FILES or f.endswith(".log.err"):
                report.reject(rel, "excluded-by-name")
                continue
            if not screen_file(p, rel, report):
                continue
            t = classify(p)
            if t is None:
                report.reject(rel, "unclassified")
                continue
            report.accept(rel)
            grouped.setdefault(t, []).append(rel)
    for t in grouped:
        grouped[t].sort()
    return grouped, report


def export():
    grouped, report = collect()
    total_files = sum(len(v) for v in grouped.values())
    print(f"Types found: {sorted(grouped)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for t, files in sorted(grouped.items()):
        out_path = OUT / f"{t}.txt"
        total_lines = 0
        with open(out_path, "w", encoding="utf-8", newline="\n") as out:
            out.write(f"===== {t.upper()} — {len(files)} files =====\n\n")
            out.write("INDEX\n-----\n")
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
        print(f"{t}.txt: {len(files)} files, {total_lines} lines, {size_mb:.2f} MB")
    manifest = write_manifest(OUT, report)
    print(f"TOTAL: {total_files} files -> {OUT}")
    print(f"GUARD: {format_summary(report)}")
    if manifest is not None:
        print(f"MANIFEST: {manifest}")
    return report


if __name__ == "__main__":
    export()
