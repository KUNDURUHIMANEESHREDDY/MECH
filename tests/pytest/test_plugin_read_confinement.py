"""A plugin worker may not read the host's files.

Writes were confined; reads were not. The hook tested

    if flags & _WRITE_FLAGS and not _within(target):
        raise CapabilityViolation(...)

so a plugin had to be *writing* to be stopped. Reading anything the host user
could read was permitted, including ``~/.ssh/id_rsa``, ``~/.config``,
credentials and browser profiles. None of that needed a network escape: the
plugin IPC channel is already authorised, so the file content came back inside
a normal dict.

These tests drive the real audit hook in a subprocess, because an audit hook
cannot be uninstalled -- installing one inside pytest would deny reads for
every later test in the same process.

The containment tests are pure-function and run in-process: the audit confirmed
that ``runner.start_remote_plugin`` tested ``assigned is False``, so a
containment result with no ``assigned`` key at all passed the check and the
worker ran unbounded. That is a pure decision, so it is tested as one.

Two placement details the probe depends on, both of which are easy to get wrong
and quietly turn a test into a no-op:

* The "secret" must not live under the temp directory. ``default_readable_roots``
  includes ``tempfile.gettempdir()``, and pytest's ``tmp_path`` is beneath it, so
  a secret in ``tmp_path`` is *readable* and the read case would report ALLOWED
  for the wrong reason.
* Nothing here touches the real ``~/.ssh``. The probe writes a decoy directory
  under the home directory instead, and ``~/.ssh`` is covered by an in-process
  test of the path filter rather than by reading or overwriting a live key.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

from backend.plugins.capabilities import (  # noqa: E402
    _is_credential_location,
    default_readable_roots,
)
from backend.plugins.runner import containment_refusal  # noqa: E402


# ── Containment must fail closed ─────────────────────────────────────────────

def test_missing_assigned_key_is_a_refusal_not_a_pass():
    """The audit's case, exactly: a result that simply omits `assigned`.

    `assign_windows_job` returns precisely this shape when the Windows Job
    Object types cannot be built, and the old `is False` check let it through.
    """
    job = {"platform": "win32", "not_enforceable": {"file_size_mb": "..."}}
    assert job.get("assigned") is None
    assert containment_refusal(job) != "", (
        "a containment result with no 'assigned' key must be treated as a "
        "refusal, not as success")


def test_explicit_false_is_a_refusal():
    assert containment_refusal({"platform": "win32", "assigned": False}) != ""


def test_confirmed_assignment_is_accepted():
    assert containment_refusal({"platform": "win32", "assigned": True}) == ""


def test_reported_error_is_a_refusal_even_when_assigned():
    job = {"platform": "win32", "assigned": True,
           "error": "SetInformationJobObject failed"}
    assert "SetInformationJobObject" in containment_refusal(job)


def test_posix_is_not_gated_on_assigned():
    """Off Windows there is no `assigned` key by design; do not break POSIX."""
    assert containment_refusal({"platform": "posix"}) == ""


# ── The allow-list itself ────────────────────────────────────────────────────

def test_home_directory_is_not_readable():
    assert _is_credential_location(Path.home().resolve()) is True


def test_credential_directories_are_not_readable():
    home = Path.home().resolve()
    for name in (".ssh", ".aws", ".gnupg", ".config", ".netrc"):
        assert _is_credential_location(home / name) is True, name


def test_the_model_cache_is_not_treated_as_a_credential():
    """`~/.cache/huggingface` is where the weights are.

    A blanket "reject dotted directories under home" rule would refuse it and
    stop every plugin from loading a model, which is why the filter is a named
    list rather than a pattern.
    """
    home = Path.home().resolve()
    assert _is_credential_location(home / ".cache") is False
    assert _is_credential_location(home / ".cache" / "huggingface") is False


def test_readable_roots_never_include_home_or_an_ancestor_of_it():
    roots = default_readable_roots()
    assert roots, "the interpreter must always be able to import something"
    home = Path.home().resolve()
    for root in roots:
        assert root != home, f"{root} is the home directory"
        assert root not in home.parents, f"{root} is an ancestor of home"


def test_the_python_installation_is_readable():
    """Otherwise no plugin could import anything at all."""
    import sysconfig

    roots = default_readable_roots()
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    assert any(root == stdlib or stdlib in root.parents for root in roots), (
        "the standard library must be readable")


# ── Behaviour, through the real hook ─────────────────────────────────────────

PROBE = textwrap.dedent(
    """
    import os
    import shutil
    import sys
    import json
    from pathlib import Path

    repo, plugin_dir, secret_dir, extra_dir, grant = sys.argv[1:6]
    sys.path.insert(0, repo)
    from backend.plugins.capabilities import (
        install_capability_hook, CapabilityViolation,
    )

    plugin = Path(plugin_dir)
    secret = Path(secret_dir)
    extra = Path(extra_dir)

    # Everything is created before the hook goes on, so that only the reads and
    # writes under test are governed by it.
    (plugin / "data.json").write_text('{"ok": true}')
    secret_file = secret / "secret.txt"
    secret_file.write_text("TOP SECRET")
    (extra / "public.txt").write_text("PUBLIC")

    link = plugin / "escape"
    have_link = True
    try:
        link.symlink_to(secret_file)
    except OSError:
        have_link = False

    readable = [grant] if grant else []
    install_capability_hook(writable_roots=[str(plugin)], readable_roots=readable)

    def attempt(label, fn):
        try:
            fn()
        except CapabilityViolation:
            print("BLOCKED\\t" + label)
        except Exception as exc:
            print("OTHER\\t" + label + "\\t" + type(exc).__name__)
        else:
            print("ALLOWED\\t" + label)

    attempt("read_own_data", lambda: (plugin / "data.json").read_text())
    attempt("import_stdlib", lambda: json.dumps({"a": 1}))
    attempt("read_secret", lambda: secret_file.read_text())
    attempt("read_secret_bytes", lambda: secret_file.read_bytes())
    attempt("os_open_secret", lambda: os.open(str(secret_file), os.O_RDONLY))
    attempt("listdir_home", lambda: os.listdir(str(Path.home())))
    attempt("listdir_secret_parent", lambda: os.listdir(str(secret)))
    attempt("scandir_secret_parent", lambda: list(os.scandir(str(secret))))
    attempt("traversal_to_secret",
            lambda: (plugin / ".." / ".." / secret.name / "secret.txt").read_text())
    if have_link:
        attempt("read_through_symlink", lambda: link.read_text())
    attempt("read_extra", lambda: (extra / "public.txt").read_text())
    attempt("write_own_dir", lambda: (plugin / "out.txt").write_text("x"))
    attempt("write_secret_dir", lambda: (secret / "evil.txt").write_text("x"))
    attempt("write_home_probe_dir", lambda: (secret / "evil2.txt").write_text("x"))
    print("HAVE_LINK\\t" + str(have_link))
    """
)


def _run_probe(grant: str, tmp_path: Path) -> dict:
    """Run the probe in a subprocess. `grant` is the readable_roots entry."""
    plugin = tmp_path / "plugin_dir"
    plugin.mkdir(parents=True, exist_ok=True)

    # Both must sit under the home directory, not under tmp_path: the temp
    # directory is readable by default, which would make both cases pass for
    # the wrong reason.
    tag = uuid.uuid4().hex[:8]
    secret = Path.home() / f"mech_probe_secret_{tag}"
    extra = Path.home() / f"mech_probe_grant_{tag}"

    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(ROOT), str(plugin), str(secret),
         str(extra), grant],
        capture_output=True, text=True, cwd=str(ROOT), timeout=180,
    )
    try:
        assert out.returncode == 0, out.stderr
        return dict(
            (parts[1], parts[0]) for parts in
            (line.split("\t") for line in out.stdout.splitlines())
            if len(parts) >= 2
        )
    finally:
        for d in (secret, extra):
            shutil_rmtree(d)


def shutil_rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def default_verdicts(tmp_path_factory) -> dict:
    return _run_probe("", tmp_path_factory.mktemp("probe"))


@pytest.mark.parametrize("case", [
    "read_secret",
    "read_secret_bytes",
    "os_open_secret",
    "listdir_home",
    "listdir_secret_parent",
    "scandir_secret_parent",
    "traversal_to_secret",
    "read_extra",
])
def test_reads_outside_the_allowlist_are_refused(default_verdicts, case):
    assert default_verdicts.get(case) == "BLOCKED", (
        f"{case} was {default_verdicts.get(case)!r}: the host's files are "
        f"still readable from a plugin worker")


@pytest.mark.parametrize("case", ["read_own_data", "import_stdlib", "write_own_dir"])
def test_legitimate_operations_still_work(default_verdicts, case):
    assert default_verdicts.get(case) == "ALLOWED", (
        f"{case} was {default_verdicts.get(case)!r}: read confinement has "
        f"broken normal plugin operation")


@pytest.mark.parametrize("case", [
    "write_secret_dir",
    "write_home_probe_dir",
])
def test_writes_outside_the_plugin_directory_are_still_refused(default_verdicts, case):
    assert default_verdicts.get(case) == "BLOCKED"


def test_symlink_escape_is_refused(default_verdicts):
    """`_normalize` resolves symlinks, so a link is not a way around the roots."""
    if default_verdicts.get("HAVE_LINK") != "True":
        pytest.skip("this host cannot create symlinks without "
                    "SeCreateSymbolicLinkPrivilege")
    assert default_verdicts["read_through_symlink"] == "BLOCKED"


def test_an_explicitly_granted_root_becomes_readable(tmp_path):
    """A declared capability is honoured -- the list is a policy, not a wall."""
    tag = uuid.uuid4().hex[:8]
    extra = Path.home() / f"mech_probe_grant_{tag}"
    try:
        extra.mkdir(parents=True, exist_ok=True)
        (extra / "public.txt").write_text("PUBLIC")
        verdicts = _run_probe(str(extra), tmp_path)
    finally:
        shutil_rmtree(extra)

    assert verdicts["read_extra"] == "ALLOWED"
    assert verdicts["read_secret"] == "BLOCKED", (
        "granting one extra root must not widen the allow-list")


def test_a_granted_credential_path_is_still_refused(tmp_path):
    """Configuration cannot be used to hand a plugin the host's credentials.

    `readable_roots` is caller-supplied, so it is exactly the parameter an
    attacker would aim at if the filter applied only to the defaults. Tested
    against the real resolution function rather than through the hook, so the
    test does not have to create a credential directory to have something to
    refuse.
    """
    from backend.plugins.capabilities import resolve_readable_roots

    ssh = (Path.home() / ".ssh").resolve()
    allowed = resolve_readable_roots([], [ssh])

    assert ssh not in allowed, (
        "readable_roots accepted the host's .ssh directory; the credential "
        "filter is not applied to caller-granted roots")


def test_a_granted_ordinary_path_is_honoured(tmp_path):
    """Positive control: the filter is not simply refusing everything."""
    from backend.plugins.capabilities import resolve_readable_roots

    granted = (Path.home() / "mech_probe_grant_ok").resolve()
    allowed = resolve_readable_roots([], [granted])
    assert granted in allowed