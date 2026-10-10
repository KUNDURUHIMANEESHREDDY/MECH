"""The source exporters must not publish secrets, and must say what they dropped.

The rule under test
-------------------
`scripts/export_all_source.py`, `export_by_area.py` and `export_by_type.py` walk
the repository and concatenate what they find into one reviewable text file.
That output is shared, attached to review requests, pasted into issues. It is a
*publication* step.

Before `scripts/export_guard.py`, the decision about what could be published was
a side effect of two unrelated decisions:

* `export_by_area.py` let any extensionless file through
  (`if f not in ("Dockerfile",) and not p.suffix == "": continue`). A file with
  no extension is exactly what `.env`, `.npmrc` and `id_rsa` look like --
  `Path(".env").suffix` is `""` -- so the rule meant to admit `Dockerfile` also
  admitted every credential file on disk.
* `export_by_type.py` classified anything unrecognised as `other`, and `other`
  was an *exported* bucket. `.json` is an exportable extension, so
  `credentials.json` had two independent routes into a shareable snapshot.

Also asserted here: that the screen accounts for every file it walks (no silent
skips), that a custom `--output` cannot ingest the previous snapshot, and that
the guard does not flag its own source -- a scanner that disables itself on its
second run is not a control.

Fake secrets in this file are built by concatenation so that this test file
itself stays publishable. `test_no_repo_file_is_a_secret` runs the real screen
over the real tree, which is what keeps that true.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pytest

from scripts import export_all_source, export_by_area, export_by_type
from scripts import export_guard

ROOT = Path(__file__).resolve().parents[2]

#: Directories the repository-wide screens skip. `frontend/node_modules` alone
#: is ~100k files, so an unscreened walk of ROOT takes minutes; these are the
#: same names the exporters themselves exclude.
REPO_SKIP_DIRS = (
    ".git", "node_modules", "__pycache__", "exports", ".venv", "venv",
    "dist", "build", ".pytest_cache", ".pytest-tmp", "coverage", ".agents",
    ".idea", ".vscode", "test-results", "screenshots", "design-demos",
    "MECH-standalone", "release", ".cache", ".claude", ".omo",
)

# Built by concatenation: see the module docstring. A literal here would make
# this file a secret-bearing file, and the repository-wide screen below would
# then exclude MECH's own test suite from every snapshot.
FAKE_PEM = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKfake\n"
FAKE_AWS = "AKIA" + "IOSFODNN7EXAMPLE"
FAKE_GITHUB = "ghp_" + "a" * 36
FAKE_SLACK = "xoxb-" + "123456789012-abcdefghijklmnop"
FAKE_GOOGLE = "AIza" + "b" * 35
FAKE_STRIPE = "sk_live_" + "c" * 24
FAKE_JWT = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
            ".dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk")
FAKE_OPENAI = "sk-" + "e" * 40


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A miniature repository containing one file per case under test."""
    (tmp_path / "backend").mkdir()
    (tmp_path / "docs").mkdir()

    (tmp_path / "backend" / "ok.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# fixture\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("all:\n\techo hi\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name": "x"}\n', encoding="utf-8")

    # name rules
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    (tmp_path / ".env.production").write_text("SECRET=1\n", encoding="utf-8")
    (tmp_path / ".npmrc").write_text("//registry:_authToken=abc\n", encoding="utf-8")
    (tmp_path / "server.pem").write_text("x", encoding="utf-8")
    (tmp_path / "tls.key").write_text("x", encoding="utf-8")
    (tmp_path / "store.jks").write_text("x", encoding="utf-8")
    (tmp_path / "id_rsa").write_text("x", encoding="utf-8")
    (tmp_path / "credentials").write_text("u:p\n", encoding="utf-8")
    (tmp_path / "credentials.json").write_text('{"user": "u"}\n', encoding="utf-8")
    (tmp_path / "service-account.json").write_text('{"type": "x"}\n', encoding="utf-8")
    (tmp_path / "access_token.json").write_text("{}n", encoding="utf-8")
    (tmp_path / "docs" / "api_secret_notes.md").write_text("hi\n", encoding="utf-8")

    # content rules -- each in an otherwise innocuous source file
    (tmp_path / "backend" / "pem_case.py").write_text(FAKE_PEM, encoding="utf-8")
    (tmp_path / "backend" / "aws_case.py").write_text(f"K = '{FAKE_AWS}'\n", encoding="utf-8")
    (tmp_path / "backend" / "gh_case.py").write_text(f"T = '{FAKE_GITHUB}'\n", encoding="utf-8")
    (tmp_path / "backend" / "slack_case.py").write_text(f"S = '{FAKE_SLACK}'\n", encoding="utf-8")
    (tmp_path / "backend" / "google_case.py").write_text(f"G = '{FAKE_GOOGLE}'\n", encoding="utf-8")
    (tmp_path / "backend" / "stripe_case.py").write_text(f"K = '{FAKE_STRIPE}'\n", encoding="utf-8")
    (tmp_path / "backend" / "jwt_case.py").write_text(f"J = '{FAKE_JWT}'\n", encoding="utf-8")
    (tmp_path / "backend" / "openai_case.py").write_text(f"K = '{FAKE_OPENAI}'\n", encoding="utf-8")

    # unknown type: must be excluded, never bucketed as `other`
    (tmp_path / "mystery.qqq").write_text("opaque bytes\n", encoding="utf-8")
    (tmp_path / "noextension").write_text("who knows\n", encoding="utf-8")

    return tmp_path


def _screen_all(root: Path) -> List[str]:
    """Relative paths of everything the guard would publish under `root`."""
    report = export_guard.ScreenReport()
    export_guard.walk_repository(root, exclude_dirs=REPO_SKIP_DIRS, report=report)
    return sorted(report.included)


def _rejected(root: Path) -> dict:
    report = export_guard.ScreenReport()
    export_guard.walk_repository(root, exclude_dirs=REPO_SKIP_DIRS, report=report)
    return {item.path: item.reason for item in report.excluded}


# ── Name rules ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", [
    ".env", ".env.production", ".npmrc", "server.pem", "tls.key", "store.jks",
    "id_rsa", "credentials", "credentials.json", "service-account.json",
    "access_token.json",
])
def test_no_credential_file_is_publishable(rel, tree):
    assert rel not in _screen_all(tree), f"{rel} would be embedded in a snapshot"


def test_credentials_json_is_rejected_even_though_json_is_exportable():
    """The `.json` case is the one with two independent routes out.

    `.json` is a legitimately exportable extension, so extension policy alone
    cannot protect it. The audit calls this out and it is the sharpest case
    here, so it gets its own assertion rather than one row of a parametrisation.
    """
    assert ".json" in export_guard.TEXT_EXTENSIONS
    assert export_guard.name_exclusion_reason("credentials.json") == "secret-name"


def test_pem_and_key_files_are_rejected():
    for name in ("a.pem", "a.key", "a.crt", "a.p12", "a.pfx", "a.jks", "a.keystore"):
        assert export_guard.name_exclusion_reason(name) == "secret-name", name


def test_secret_word_inside_a_non_source_filename_is_rejected(tree):
    assert "docs/api_secret_notes.md" not in _screen_all(tree)


def test_secret_word_inside_a_source_filename_is_not_ground_for_rejection():
    """`token_inspector.py` is code.

    A bare substring rule would have excluded
    `backend/interpretability/inspectors/token.py` and
    `frontend/src/design/tokens/` from every snapshot -- hiding source to
    protect against a leak that cannot occur in a source file, because the
    content scan is what covers those.
    """
    for name in ("token_inspector.py", "tokens.py", "secret_manager.ts",
                 "private_key_helper.ts"):
        assert export_guard.name_exclusion_reason(name) is None, name


def test_the_repository_token_modules_are_actually_publishable():
    """Pinned against the real tree, so the rule above cannot be narrowed later."""
    published = _screen_all(ROOT)
    for rel in ("scripts/export_guard.py", "scripts/export_all_source.py",
                "backend/interpretability/inspectors/token.py"):
        assert rel in published, f"{rel} is excluded; the name rules are too broad"


# ── Content rules ───────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", [
    "backend/pem_case.py", "backend/aws_case.py", "backend/gh_case.py",
    "backend/slack_case.py", "backend/google_case.py", "backend/stripe_case.py",
    "backend/jwt_case.py", "backend/openai_case.py",
])
def test_no_secret_bearing_file_is_publishable(rel, tree):
    assert rel not in _screen_all(tree)


def test_each_content_rule_names_itself_in_the_rejection_reason(tree):
    reasons = _rejected(tree)
    for rel, rule in [
        ("backend/pem_case.py", "pem-private-key"),
        ("backend/aws_case.py", "aws-access-key-id"),
        ("backend/gh_case.py", "github-token"),
        ("backend/slack_case.py", "slack-token"),
        ("backend/google_case.py", "google-api-key"),
        ("backend/stripe_case.py", "stripe-live-key"),
        ("backend/jwt_case.py", "json-web-token"),
        ("backend/openai_case.py", "openai-style-key"),
    ]:
        assert reasons[rel].startswith("secret-content:"), rel
        assert rule in reasons[rel], f"{rel} was rejected without naming {rule}"


def test_ordinary_source_is_not_mistaken_for_a_secret():
    """The false-positive side.

    Every rule is high-confidence *and* narrow. A rule that matches ordinary
    code gets disabled by whoever hits it first, and then the real leak ships.
    """
    benign = (
        "import os\n"
        "TOKEN_PRICE = 3.5\n"
        "def get_token():\n"
        "    return os.environ['TOKEN']\n"
        "# password hashing uses bcrypt\n"
        "API_KEY = os.environ['API_KEY']\n"
        "jwt = 'eyJ-not-a-jwt'\n"
    )
    assert export_guard.content_exclusion_reasons(benign) == []


def test_the_guard_does_not_flag_its_own_source():
    """A scanner that disables itself on the second run is not a control.

    Every rule's pattern is written so its own literal fails to match it. This
    is what keeps `scripts/export_guard.py` -- and this test file, whose fake
    secrets are built by concatenation for the same reason -- publishable.
    """
    own = (ROOT / "scripts" / "export_guard.py").read_text(encoding="utf-8")
    assert export_guard.content_exclusion_reasons(own) == []
    assert export_guard.name_exclusion_reason("export_guard.py") is None


def test_no_repo_file_is_a_secret():
    """The real tree, screened with the real rules.

    This is the check that keeps the guard honest in both directions: a new
    credential committed to MECH fails here, and so does a test fixture that
    stopped hiding its fake secret.
    """
    report = export_guard.ScreenReport()
    export_guard.walk_repository(ROOT, exclude_dirs=REPO_SKIP_DIRS, report=report)
    offenders = [f"{i.path}: {i.reason}" for i in report.excluded
                 if i.reason.startswith("secret-content")]
    assert offenders == [], (
        "a tracked file trips a secret rule:\n  " + "\n  ".join(offenders))


# ── Unknown types ───────────────────────────────────────────────────────

def test_unknown_file_types_are_excluded_not_bucketed(tree):
    published = _screen_all(tree)
    assert "mystery.qqq" not in published
    assert "noextension" not in published


def test_the_other_bucket_no_longer_exists():
    """`classify` returning `other` is how unknown bytes became published bytes."""
    src = (ROOT / "scripts" / "export_by_type.py").read_text(encoding="utf-8")
    assert '"other"' not in src, (
        "export_by_type.py reintroduced an 'other' bucket; unknown files must be "
        "excluded, not published under a catch-all name")
    assert "unclassified" in src, (
        "the unclassifiable path is now silently skipped instead of recorded")


def test_recognised_project_files_are_publishable(tree):
    """Excluding unknown types must not exclude the files that were always meant
    to be here -- `Makefile`, `package.json`, `.gitignore`."""
    published = _screen_all(tree)
    for rel in ("backend/ok.py", "README.md", "Makefile", "package.json"):
        assert rel in published, f"{rel} is no longer publishable"


# ── Accounting ──────────────────────────────────────────────────────────

def test_every_walked_file_is_either_published_or_explained(tree):
    """No silent skips.

    The audit's point on the size limit: `TOTAL: N files` does not mean "all
    source files" when nothing says what was dropped. The report has to account
    for every path the walk visited.
    """
    report = export_guard.ScreenReport()
    export_guard.walk_repository(tree, exclude_dirs=REPO_SKIP_DIRS, report=report)

    visited = {p.relative_to(tree).as_posix()
               for p in tree.rglob("*") if p.is_file()}
    accounted = set(report.included) | {i.path for i in report.excluded}

    assert visited == accounted, (
        "files walked but neither published nor explained: "
        f"{sorted(visited - accounted)}")
    assert report.included_count == len(report.included)


def test_the_manifest_counts_add_up(tree):
    report = export_guard.ScreenReport()
    export_guard.walk_repository(tree, exclude_dirs=REPO_SKIP_DIRS, report=report)

    manifest = report.as_manifest()

    assert manifest["included"] == len(report.included)
    assert manifest["excluded"] == len(report.excluded)
    assert sum(manifest["excluded_reasons"].values()) == manifest["excluded"]
    assert len(manifest["excluded_files"]) == manifest["excluded"]


def test_the_manifest_is_written_as_json(tree, tmp_path):
    report = export_guard.ScreenReport()
    export_guard.walk_repository(tree, exclude_dirs=REPO_SKIP_DIRS, report=report)

    path = export_guard.write_manifest(tmp_path, report)

    assert path is not None and path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == report.as_manifest()


def test_a_binary_is_reported_as_such_not_as_unknown(tree):
    """`binary-or-nontext` and `unrecognized-type` are different statements.

    "We recognised this and skipped it" and "we do not know what this is" should
    not share a reason code, or a manifest reader cannot tell a deliberate skip
    from a gap in coverage.
    """
    (tree / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    reasons = _rejected(tree)
    assert reasons["logo.png"] == "binary-or-nontext"


def test_an_oversized_file_is_reported_not_silently_dropped(tree):
    big = tree / "huge.json"
    big.write_text("x" * (export_guard.MAX_CONTENT_BYTES + 10), encoding="utf-8")
    try:
        reasons = _rejected(tree)
    finally:
        big.unlink()
    assert reasons["huge.json"] == "too-large"


# ── The exporters themselves ────────────────────────────────────────────

def test_export_all_source_excludes_its_own_custom_output(tmp_path: Path,
                                                          monkeypatch):
    """Self-ingestion.

    `EXCLUDE_FILES` names `MECH_all_source.txt` and nothing else, so
    `--output snapshot.txt` was discovered as a root `.txt` source file and
    embedded in the next run's snapshot. The resolved output path is now
    excluded unconditionally.
    """
    monkeypatch.setattr(export_all_source, "ROOT", tmp_path)
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "ok.py").write_text("x = 1\n", encoding="utf-8")
    output = tmp_path / "snapshot.txt"
    output.write_text("PREVIOUS SNAPSHOT CONTENT\n", encoding="utf-8")
    assert output.name not in export_all_source.EXCLUDE_FILES, (
        "this test asserts against the name list; if it gained the name, the "
        "defect it describes is gone and this test needs replacing")

    report = export_guard.ScreenReport()
    collected = export_all_source.collect_files(report, {output.resolve()})

    assert "snapshot.txt" not in collected
    assert report.reason_counts().get("exporter-output", 0) >= 1


def test_export_all_source_walks_the_whole_tree(tmp_path: Path, monkeypatch):
    """The completeness claim in the filename.

    The root scan accepted only `.py/.js/.ini/.txt` and the CI scan named
    exactly one workflow, so `package.json`, `pyproject.toml`, `Makefile` and
    every other workflow were missing from a file headed "Complete Source Code
    Repository".
    """
    monkeypatch.setattr(export_all_source, "ROOT", tmp_path)
    (tmp_path / "backend").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "frontend" / "src").mkdir(parents=True)
    (tmp_path / ".github" / "workflows").mkdir(parents=True)

    (tmp_path / "backend" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    (tmp_path / "frontend" / "src" / "App.vue").write_text("<template/>\n", encoding="utf-8")
    (tmp_path / "frontend" / "package.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "package.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "Makefile").write_text("all:\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text("name: ci\n", encoding="utf-8")
    (tmp_path / ".github" / "workflows" / "nightly.yml").write_text("name: nightly\n", encoding="utf-8")

    report = export_guard.ScreenReport()
    collected = set(export_all_source.collect_files(report, set()))

    for rel in ("backend/a.py", "tests/test_a.py", "frontend/src/App.vue",
                "frontend/package.json", "pyproject.toml", "package.json",
                "Makefile", ".github/workflows/ci.yml",
                ".github/workflows/nightly.yml"):
        assert rel in collected, f"{rel} is missing from the 'complete' export"


def test_export_all_source_publishes_no_secret(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(export_all_source, "ROOT", tmp_path)
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "ok.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=" + "z" * 40 + "\n", encoding="utf-8")
    (tmp_path / "leaky.py").write_text(f"K = '{FAKE_AWS}'\n", encoding="utf-8")

    out = tmp_path / "out.txt"
    count, _lines, report = export_all_source.export_all(out)

    assert count == 1
    body = out.read_text(encoding="utf-8")
    assert ".env" not in body
    assert "leaky.py" not in body
    assert "IOSFODNN7EXAMPLE" not in body
    assert report.reason_counts().get("secret-name", 0) >= 1


def test_export_by_area_publishes_no_secret(tree, monkeypatch):
    monkeypatch.setattr(export_by_area, "ROOT", tree)
    monkeypatch.setattr(export_by_area, "OUT", tree / "exports" / "code_by_area")

    report = export_by_area.export()

    for excluded in (".env", ".npmrc", "server.pem", "id_rsa",
                     "credentials.json", "service-account.json",
                     "backend/pem_case.py", "backend/aws_case.py"):
        assert excluded in {i.path for i in report.excluded}, excluded

    for area_file in (tree / "exports" / "code_by_area").glob("*.txt"):
        body = area_file.read_text(encoding="utf-8")
        assert "IOSFODNN7EXAMPLE" not in body
        assert "BEGIN RSA PRIVATE KEY" not in body
        assert "_authToken" not in body


def test_export_by_type_publishes_no_secret_and_no_other_bucket(tree, monkeypatch):
    monkeypatch.setattr(export_by_type, "ROOT", tree)
    monkeypatch.setattr(export_by_type, "OUT", tree / "exports" / "code_by_type")

    report = export_by_type.export()

    rejected = {i.path for i in report.excluded}
    for excluded in (".env", "credentials.json", "backend/jwt_case.py",
                     "mystery.qqq", "noextension"):
        assert excluded in rejected, excluded

    assert not (tree / "exports" / "code_by_type" / "other.txt").exists(), (
        "an 'other' bucket was written again; unknown files must be excluded")

    for type_file in (tree / "exports" / "code_by_type").glob("*.txt"):
        assert "eyJhbGciOi" not in type_file.read_text(encoding="utf-8")


def test_export_by_type_still_classifies_the_usual_suspects(tree, monkeypatch):
    monkeypatch.setattr(export_by_type, "ROOT", tree)
    grouped, _report = export_by_type.collect()

    assert "backend/ok.py" in grouped["python"]
    assert "README.md" in grouped["markdown"]
    assert "Makefile" in grouped["config"]
    assert "package.json" in grouped["json"]


def test_a_secret_in_an_encoded_shape_is_still_excluded(tmp_path):
    """A UTF-16 private key must not be published as an ordinary text file.

    The content scan decodes as UTF-8 with `errors="replace"`, so a secret saved
    in UTF-16 -- Notepad's historical default for files with unusual characters
    -- is mangled into NUL bytes and clears every regex, then is published with
    the key recoverable by stripping NULs. Found by an independent review, which
    executed the guard over 1,446 real files rather than reasoning about it.

    Verified before fixing: a UTF-16LE PEM in `notes.txt` was `accepted = True`.

    `FAKE_PEM` is built by concatenation -- see the module docstring: a literal
    here would make this file itself a secret-bearing source file.
    """
    import scripts.export_guard as guard

    utf16 = tmp_path / "notes.txt"
    utf16.write_bytes(FAKE_PEM.encode("utf-16-le"))
    report = guard.ScreenReport()

    assert guard.screen_file(utf16, tmp_path, report) is False, (
        "a private key stored as UTF-16 was accepted for publication")
    reasons = report.reason_counts()
    assert any("secret-content-encoding" in k for k in reasons), reasons
    # The same bytes as UTF-8 must still be reported by the plain rule, so the
    # encoding reason is a *second* net rather than a replacement for the first.
    plain = tmp_path / "plain.txt"
    plain.write_text(FAKE_PEM, encoding="utf-8")
    plain_report = guard.ScreenReport()
    assert guard.screen_file(plain, tmp_path, plain_report) is False
    assert not any("secret-content-encoding" in k
                   for k in plain_report.reason_counts()), (
        "the encoding reason must apply only to the encoded case")


def test_an_unquoted_aws_secret_access_key_is_excluded(tmp_path):
    """`~/.aws/credentials` is an INI whose values are unquoted.

    The pattern required quotes on both sides, so it published exactly the file
    it was written to catch. Easy to miss because the access-key-id rule does
    fire when both keys are in the same file, which they usually are.

    Note this test builds the *secret value* only, not an `AKIA...` access-key
    id: the id rule is a different net, and including both would make this pass
    for the wrong reason.
    """
    import scripts.export_guard as guard

    ini = tmp_path / "cfg.ini"
    ini.write_text(
        "[default]\naws_secret_access_key = "
        + "wJalrXUtnFEMI" + "/K7MDENG" + "/bPxRfiCY" + "EXAMPLEKEY" + "\n",
        encoding="utf-8")
    report = guard.ScreenReport()

    assert guard.screen_file(ini, tmp_path, report) is False
    reasons = report.reason_counts()
    assert any("aws-secret-access-key" in k for k in reasons), reasons


def test_the_quoted_form_is_still_excluded(tmp_path):
    """The control: the original rule still holds after making quotes optional."""
    import scripts.export_guard as guard

    ini = tmp_path / "cfg.ini"
    ini.write_text(
        '[default]\naws_secret_access_key = "'
        + "wJalrXUtnFEMI" + "/K7MDENG" + "/bPxRfiCY" + "EXAMPLEKEY" + '"\n',
        encoding="utf-8")
    report = guard.ScreenReport()

    assert guard.screen_file(ini, tmp_path, report) is False
    reasons = report.reason_counts()
    assert any("aws-secret-access-key" in k for k in reasons), reasons
