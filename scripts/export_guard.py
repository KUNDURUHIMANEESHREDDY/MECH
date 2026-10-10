"""Secret and type screening for the source-export scripts.

Why this module exists
----------------------
`export_by_area.py`, `export_by_type.py` and `export_all_source.py` walk the
repository and concatenate what they find into a single reviewable text file.
That output gets shared, attached to review requests, and pasted into issues --
which makes it a *publication* step, not a build step.

Before this module, the boundary that decided what could be published was a
side effect of two unrelated decisions:

* `export_by_area.py` allowed *any* extensionless file through
  (`if f not in ("Dockerfile",) and not p.suffix == "": continue`). A file with
  no extension is exactly what `.env`, `.npmrc`, `credentials` and `id_rsa` look
  like -- `Path(".env").suffix` is the empty string -- so the rule that admitted
  them was the same rule meant to admit `Dockerfile`.
* `export_by_type.py` classified anything it did not recognise as `other`, and
  `other` is an *exported* bucket. `.json` is an exportable extension, so a
  `credentials.json` was exported by two independent routes.

A shared allow-list of extensions does not protect a file's contents. Both
`backend/plugins/capabilities.py` (which stops plugins reading `$HOME/.aws`) and
this repository's `.gitignore` (which lists `*.pem` and `*.key` as a "second
line of defence") treat a conventional secret path as a *hint*, not a control.
The control here is the pair: deny by name, then scan what survives by content.

The policy in one paragraph
---------------------------
A file is published only if its name is not a known credential path AND its
contents match no high-confidence secret pattern AND its type is one the
exporter recognises. Anything else is recorded in a manifest with the reason,
so the summary line counts what was published rather than claiming it covered
the tree. Unknown types are *excluded*, never bucketed as `other` -- an
unrecognised file is unexamined, and publishing unexamined bytes is the failure
mode this module exists to remove.

Design notes that look odd but are deliberate
---------------------------------------------
* **The patterns do not match this file.** Every rule above is written so that
  its own source literal fails to match it (a character class never contains its
  own bracket). A scanner that flags itself is a scanner that disables itself on
  the second run. `test_the_guard_does_not_flag_its_own_source` pins this.
* **The tests build fake secrets by concatenation.** For the same reason:
  `tests/pytest/test_export_secret_exclusion.py` must be publishable itself.
* **Filename rules are type-aware.** A module named `token_inspector.py` is
  code; a `token.json` is a credential. Denial by bare substring would have
  excluded `backend/interpretability/inspectors/token.py` and
  `frontend/src/design/tokens/`, so substring rules apply to non-source files
  and to exact basenames only.
* **Nothing here rewrites secrets out of text.** Detection is high-confidence by
  design; a redactor that guesses is a redactor that silently corrupts the
  snapshot it is supposed to protect. The file is dropped and reported instead.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

# Files larger than this are generated blobs (bundles, lockfiles, snapshots) and
# are reported as skipped rather than silently dropped.
MAX_CONTENT_BYTES = 2_000_000

# --------------------------------------------------------------------------
# Extension policy
# --------------------------------------------------------------------------

#: Extensions whose contents are human-written source. A sensitive word in the
#: *name* of one of these is far more likely to be a module name than a
#: credential -- `token_inspector.py`, `design/tokens/colors.ts` -- so the
#: substring name rules below do not apply to them.
#: Extensions whose contents are human-written source code. A sensitive word in
#: the *name* of one of these is far more likely to be a module name than a
#: credential -- `token_inspector.py`, `design/tokens/colors.ts` -- so the
#: substring name rules below do not apply to them.
#:
#: Prose (`.md`, `.txt`, `.rst`) is deliberately *not* in this set. A document
#: called `api_secret_notes.md` is exactly where someone pastes a working key.
SOURCE_CODE_EXTENSIONS = frozenset({
    ".py", ".pyi", ".js", ".cjs", ".mjs", ".jsx", ".ts", ".tsx", ".mts", ".cts",
    ".vue", ".svelte", ".go", ".rs", ".java", ".kt", ".rb", ".php", ".c", ".h",
    ".cc", ".cpp", ".hpp", ".cs", ".swift", ".scala", ".sh", ".bash", ".zsh",
    ".ps1", ".bat", ".cmd", ".sql", ".html", ".htm", ".css", ".scss", ".less",
})

#: Extensions that are never published regardless of name: binaries, archives,
#: media, and build output.
EXCLUDED_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".exe", ".dll", ".so", ".dylib", ".pak", ".bin", ".zip", ".tar", ".gz",
    ".bz2", ".7z", ".asar", ".db", ".sqlite", ".sqlite3", ".pyc", ".pyo",
    ".pyd", ".blockmap", ".dat", ".patch", ".pdf", ".woff", ".woff2", ".ttf",
    ".eot", ".otf", ".mp4", ".mov", ".avi", ".mp3", ".wav", ".pack", ".idx",
})

#: Extensions the exporters recognise as publishable text. Anything else is
#: excluded as `unrecognized-type` rather than bucketed.
TEXT_EXTENSIONS = frozenset({
    ".py", ".js", ".cjs", ".mjs", ".jsx", ".ts", ".mts", ".cts", ".tsx",
    ".vue", ".css", ".scss", ".less", ".html", ".htm",
    ".json", ".jsonc", ".jsonl", ".md", ".mdx", ".rst", ".txt",
    ".yml", ".yaml", ".ini", ".toml", ".cfg", ".conf", ".properties",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".sql",
})

#: Dotfiles with no extension carry an empty `Path.suffix`, so they are matched
#: by name in `BARE_FILENAMES` rather than listed here. `.env` is likewise a
#: name rule, not an extension rule -- see `DENY_NAME_PREFIXES`.
TEXT_DOTFILES = frozenset({
    ".gitignore", ".gitattributes", ".dockerignore", ".editorconfig",
    ".prettierrc", ".prettierignore",
})

#: Extensionless files that are genuine repository content. An allow-list, not a
#: pass-through: the previous "no extension means include it" rule is what let
#: `.env` and `.npmrc` into a shareable snapshot.
BARE_FILENAMES = frozenset({
    "Dockerfile", "Makefile", "LICENSE", "LICENCE", "NOTICE", "COPYING",
    "CODEOWNERS", "Gemfile", "Procfile",
    # Dotfiles have an empty `Path.suffix`, so they belong here rather than in
    # `TEXT_EXTENSIONS`. `.npmrc` is absent on purpose: it is a credential.
}) | TEXT_DOTFILES

# --------------------------------------------------------------------------
# Name policy
# --------------------------------------------------------------------------

#: Prefix rules on the file name. Checked before everything else so that
#: `.env.production` cannot be smuggled past by an unexpected extension.
DENY_NAME_PREFIXES = (".env",)

#: Exact-name rules, matched on the lowercased file name.
DENY_EXACT_NAMES = frozenset({
    ".npmrc", ".pypirc", ".netrc", ".git-credentials", ".htpasswd",
    ".dockercfg", "credentials", "credentials.json", "secrets.json",
    "service-account.json", "serviceaccount.json",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
})

#: Extension rules for container/keystore formats that hold private material.
DENY_EXTENSIONS = frozenset({
    ".pem", ".key", ".crt", ".cer", ".der", ".p12", ".pfx", ".jks",
    ".keystore", ".ppk", ".asc", ".gpg", ".kdbx", ".p8", ".jceks",
})

#: Exact *basenames* (extension stripped) that are credentials whatever their
#: extension. ``token`` is deliberately absent: ``backend/interpretability/
#: inspectors/token.py`` is a real module in this repository, and denying it
#: would hide source code to protect against a leak that cannot occur there --
#: the content scan below is what covers source files.
DENY_BASENAMES = frozenset({
    "credential", "credentials", "secret", "secrets", "password", "passwords",
    "passwd", "apikey", "api_key", "private_key", "privatekey",
    "service_account", "serviceaccount", "htpasswd",
})

#: Substrings that mark a *non-source* file as credential material. Applied only
#: where `extension not in SOURCE_CODE_EXTENSIONS`; see the module docstring.
DENY_NAME_SUBSTRINGS = (
    "credential", "secret", "password", "passwd", "passphrase",
    "apikey", "api-key", "api_key", "private_key", "privatekey",
    "private-key", "service-account", "service_account",
    "token", "auth_token", "access_token",
)

# --------------------------------------------------------------------------
# Content policy
# --------------------------------------------------------------------------

#: High-confidence secret shapes. Deliberately not exhaustive: a rule that
#: matches ordinary source is a rule that gets disabled. Each entry is
#: ``(name, pattern)`` and each pattern is written so its own literal in this
#: file fails to match it.
SECRET_CONTENT_RULES: Sequence[tuple] = (
    ("pem-private-key",
     r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"),
    ("private-key-json-field",
     r"(?i)\"private_?key\"\s*:\s*\"-----BEGIN"),
    ("aws-access-key-id", r"\bAKIA[0-9A-Z]{16}\b"),
    ("aws-secret-access-key",
     # Quotes are optional, not required. `~/.aws/credentials` is an INI whose
     # values are unquoted, so a rule demanding quotes on both sides published
     # exactly the file the rule was written for. It was easy to miss because
     # the access-key-id rule above does catch the same file when both keys are
     # present, which they usually are -- so the gap only shows on a file
     # carrying the secret alone.
     r"(?i)aws_?secret_?access_?key\s*[=:]\s*[\"']?[A-Za-z0-9/+=]{40}[\"']?"),
    ("github-token", r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    ("github-fine-grained-pat", r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    ("slack-token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    ("google-api-key", r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ("stripe-live-key", r"\b[rs]k_live_[A-Za-z0-9]{20,}\b"),
    ("anthropic-key", r"\bsk-ant-[A-Za-z0-9_-]{24,}\b"),
    ("openai-style-key", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
    ("json-web-token",
     r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
)

_COMPILED_CONTENT_RULES = tuple(
    (name, re.compile(pattern)) for name, pattern in SECRET_CONTENT_RULES
)


# --------------------------------------------------------------------------
# Screening results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Exclusion:
    """A file that was not published, and the single reason why."""

    path: str
    reason: str


@dataclass
class ScreenReport:
    """What the exporters published and what they did not.

    The whole point of the report is that `included` is a count of files whose
    bytes were actually read and checked, so a reader can tell the difference
    between "the tree has 428 source files" and "we looked at 428 files and
    declined to publish 17 more".
    """

    included: List[str] = field(default_factory=list)
    excluded: List[Exclusion] = field(default_factory=list)

    def accept(self, rel: str) -> None:
        self.included.append(rel)

    def reject(self, rel: str, reason: str) -> None:
        self.excluded.append(Exclusion(path=rel, reason=reason))

    @property
    def included_count(self) -> int:
        return len(self.included)

    def reason_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.excluded:
            counts[item.reason] = counts.get(item.reason, 0) + 1
        return dict(sorted(counts.items()))

    def as_manifest(self) -> Dict[str, object]:
        return {
            "included": self.included_count,
            "excluded": len(self.excluded),
            "excluded_reasons": self.reason_counts(),
            "excluded_files": [
                {"path": item.path, "reason": item.reason}
                for item in sorted(self.excluded, key=lambda e: e.path)
            ],
        }


# --------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------

def name_exclusion_reason(name: str) -> Optional[str]:
    """Why `name` must never be published, or ``None`` if it is acceptable.

    Name-only, and deliberately the cheap first pass: the content scan is the
    expensive one and should not run for a file this already rejects.
    """
    lowered = name.lower()
    extension = Path(lowered).suffix

    for prefix in DENY_NAME_PREFIXES:
        if lowered.startswith(prefix):
            return "secret-name"
    if lowered in DENY_EXACT_NAMES:
        return "secret-name"
    if extension in DENY_EXTENSIONS:
        return "secret-name"

    basename = Path(lowered).stem if extension else lowered
    if basename in DENY_BASENAMES:
        return "secret-name"

    # Substring rules only reach non-source files. `token_inspector.py` and
    # `design/tokens/colors.ts` are source; `token.json` is not.
    if extension not in SOURCE_CODE_EXTENSIONS:
        for needle in DENY_NAME_SUBSTRINGS:
            if needle in lowered:
                return "secret-name"
    return None


def content_exclusion_reasons(text: str) -> List[str]:
    """Names of every high-confidence secret rule `text` trips."""
    return [name for name, pattern in _COMPILED_CONTENT_RULES if pattern.search(text)]


def read_text(path: Path) -> Optional[str]:
    """Read `path` as text, or return ``None`` if it is not readable text."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def file_size(path: Path) -> Optional[int]:
    """Size in bytes, or ``None`` when the file cannot be stat'd."""
    try:
        return path.stat().st_size
    except OSError:
        return None


def is_recognised_type(name: str) -> bool:
    """Whether the exporters know what to do with a file of this type.

    Unknown is *not* publishable. The previous behaviour classified unknown
    files into an exported `other` bucket, which published bytes nobody had
    looked at.
    """
    path = Path(name)
    extension = path.suffix.lower()
    if extension in EXCLUDED_EXTENSIONS:
        return False
    if extension in TEXT_EXTENSIONS:
        return True
    return path.name in BARE_FILENAMES


def screen_file(path: Path, rel: str, report: ScreenReport,
                scan_content: bool = True) -> bool:
    """Apply every rule to one file, recording each rejection on `report`.

    Returns ``True`` when the file may be published. A ``False`` return always
    has a matching `Exclusion`, so a summary can never claim a count the
    manifest cannot account for.

    Acceptance is the caller's to record, not this function's: a caller may
    still decline a screened file for a reason of its own (an exporter that
    cannot classify it, say), and two `accept` calls for one file would inflate
    the published count.
    """
    reason = name_exclusion_reason(path.name)
    if reason is not None:
        report.reject(rel, reason)
        return False

    extension = path.suffix.lower()
    if extension in EXCLUDED_EXTENSIONS:
        # A distinct reason from `unrecognized-type`: a binary was recognised
        # and deliberately skipped, which is a different statement from "we do
        # not know what this file is".
        report.reject(rel, "binary-or-nontext")
        return False
    if not is_recognised_type(path.name):
        report.reject(rel, "unrecognized-type")
        return False

    size = file_size(path)
    if size is None:
        report.reject(rel, "unreadable")
        return False
    if size > MAX_CONTENT_BYTES:
        report.reject(rel, "too-large")
        return False

    if scan_content:
        text = read_text(path)
        if text is None:
            report.reject(rel, "unreadable")
            return False
        hits = content_exclusion_reasons(text)
        if hits:
            report.reject(rel, "secret-content:" + ",".join(sorted(hits)))
            return False

        # Encoding is not a hiding place. The scan above decodes as UTF-8; a
        # secret saved in another encoding -- UTF-16LE is Notepad's historical
        # default for files with unusual characters -- is mangled into NUL
        # bytes and clears every regex above, and is then published with the
        # key trivially recoverable by stripping the NULs.
        #
        # Verified before being fixed: a UTF-16LE PEM in `notes.txt` sailed
        # through the name rules and the type allow-list and was exported.
        obfuscated = content_exclusion_reasons(_deobfuscate(text))
        if obfuscated:
            report.reject(rel, "secret-content-encoding:"
                          + ",".join(sorted(obfuscated)))
            return False

    return True


def _deobfuscate(text: str) -> str:
    """`text` with NUL bytes removed and with any UTF-16 shapes undone.

    Only used as a *second* scan after the strict one has passed, so it can
    never turn an accept into a reject on a file the rules already understood;
    it exists to catch a secret that survived by being encoded rather than by
    being absent. Returns "" when there is nothing to undo, so the caller treats
    an all-NUL file as empty rather than scanning it twice.
    """
    if not text or "\x00" not in text:
        return ""
    stripped = text.replace("\x00", "")
    if not stripped:
        return ""
    # A UTF-16 decode of the original bytes would be the principled recovery,
    # but the caller has already decoded the file as text with errors="replace",
    # so the original bytes are gone. NUL-stripping recovers the same plaintext
    # for both UTF-16LE and UTF-16BE with an even/odd split, which is the case
    # that matters; anything genuinely binary is rejected elsewhere.
    try:
        raw = text.encode("utf-8", errors="replace")
        for codec in ("utf-16-le", "utf-16-be"):
            try:
                decoded = raw.decode(codec, errors="ignore")
            except (UnicodeDecodeError, LookupError):
                continue
            if decoded and len(decoded) < len(stripped):
                continue
    except Exception:  # pragma: no cover - defensive only
        pass
    return stripped


def walk_repository(root: Path, exclude_dirs: Sequence[str],
                    report: Optional[ScreenReport] = None,
                    exclude_paths: Optional[Set[Path]] = None,
                    scan_content: bool = True) -> ScreenReport:
    """Screen every file under `root`, honouring `exclude_dirs`.

    `exclude_paths` holds files excluded regardless of policy -- the resolved
    output path of the exporter itself, most importantly, so a snapshot cannot
    ingest the snapshot it wrote last time.
    """
    report = report if report is not None else ScreenReport()
    skipped = exclude_paths or set()
    blocked = {name.lower() for name in exclude_dirs}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames
            if d.lower() not in blocked and not d.startswith(".pytest-tmp")
        )
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            rel = path.relative_to(root).as_posix()
            if path in skipped:
                report.reject(rel, "exporter-output")
                continue
            if screen_file(path, rel, report, scan_content=scan_content):
                report.accept(rel)
    return report


def write_manifest(directory: Path, report: ScreenReport,
                   prefix: str = "export") -> Optional[Path]:
    """Write `report` as JSON next to the exports. Returns the path written."""
    import json

    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{prefix}_manifest.json"
        path.write_text(
            json.dumps(report.as_manifest(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return path
    except OSError:
        return None


def format_summary(report: ScreenReport) -> str:
    """One honest line: what was published, and what was declined and why."""
    parts = [f"published {report.included_count} files"]
    counts = report.reason_counts()
    if counts:
        detail = ", ".join(f"{reason}={count}" for reason, count in counts.items())
        parts.append(f"excluded {len(report.excluded)} ({detail})")
    else:
        parts.append("excluded 0")
    return "; ".join(parts)
