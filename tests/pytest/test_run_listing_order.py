"""Persisted runs must be listed by `created_at`, not by filename.

`list_run_records` sorted filenames descending and called the result "newest
first". But a run's filename is its run_id, and run ids are minted as
`"r" + uuid4().hex[:12]` -- twelve random hex characters with no time in them.
Reverse-sorting them is reverse-alphabetising random strings, so the listing
was in arbitrary order while its docstring promised chronology.

Two more consequences fell out of the same mistake:

  * the summary's `created` field read `publication.published_at`, which is
    empty for any run not yet published -- so the one field a caller would
    display or sort by was blank for unpublished runs;
  * the 200-entry cap was applied to the filename-sorted list, so a directory
    holding more than 200 runs kept an arbitrary 200 rather than the newest.

The envelope carries a top-level `created_at` that the Ed25519 signature binds
via `_signing_payload(run_id, record, created_at=..., public_key=...)`, so it
is the authoritative creation time: editing it breaks verification.

Every test below deliberately assigns run ids whose alphabetical order is the
OPPOSE of their creation order, so any implementation that falls back to
filename sorting fails rather than passing by luck.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core import evidence_graph as eg  # noqa: E402


def _write_run(directory, run_id, created_at, published_at=None):
    """Write an envelope with the same shape `save_run_record` produces."""
    publication = {"steps_completed": "1/1"}
    if published_at is not None:
        publication["published_at"] = published_at
    record = {"run_id": run_id, "status": "completed",
              "publication": publication}
    envelope = {
        "schema_version": eg.ENVELOPE_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": created_at,
        "record": record,
        "attestation": {"attested": False, "signature": "missing",
                        "reason": "unsigned"},
    }
    (directory / f"{run_id}.json").write_text(json.dumps(envelope),
                                             encoding="utf-8")


def _ids(directory):
    return [s["run_id"] for s in eg.list_run_records(str(directory))]


# ── ordering follows created_at ──────────────────────────────────────── #

def test_filenames_carry_no_time_so_only_created_at_can_order(tmp_path):
    """The pin: reverse-sorting these filenames gives the WRONG answer.

    `rmmm...` is created first and `raaa...` second, but `raaa...` sorts
    *after* `rmmm...` in reverse filename order -- so a filename sort lists
    the oldest run first.
    """
    _write_run(tmp_path, "rmmmmmmmmmmmm", "2024-01-01T00:00:00+00:00")
    _write_run(tmp_path, "raaaaaaaaaaaa", "2024-02-01T00:00:00+00:00")

    by_filename = sorted(("rmmmmmmmmmmmm.json", "raaaaaaaaaaaa.json"),
                         reverse=True)
    assert by_filename[0] == "rmmmmmmmmmmmm.json", (
        "premise check: filename order must disagree with time order")

    assert _ids(tmp_path) == ["raaaaaaaaaaaa", "rmmmmmmmmmmmm"], (
        "the listing followed the filename order instead of created_at")


@pytest.mark.parametrize("count", [2, 3, 5, 9])
def test_many_runs_are_listed_in_exact_reverse_creation_order(tmp_path, count):
    """Ids descend as time ascends, so filename sort is exactly inverted."""
    written = []
    for i in range(count):
        # Older runs get alphabetically LATER ids.
        run_id = f"r{chr(ord('z') - i)}{'q' * 11}"
        created = f"2024-01-{i + 1:02d}T00:00:00+00:00"
        _write_run(tmp_path, run_id, created)
        written.append((run_id, created))

    expected = [run_id for run_id, _ in sorted(written,
                                               key=lambda p: p[1], reverse=True)]
    # Premise: alphabetical ascending order is newest-first here, so any
    # implementation that sorted by filename would produce this same answer
    # in reverse and fail.
    assert expected == sorted(run_id for run_id, _ in written), (
        "premise check: alphabetical order must already be newest-first")
    assert _ids(tmp_path) == expected, (
        f"{count} runs were not listed in reverse creation order")


# ── the summary reports creation time, not publication time ──────────── #

def test_created_reports_created_at_not_published_at(tmp_path):
    _write_run(tmp_path, "runone", "2024-06-01T12:00:00+00:00",
               published_at="2024-06-09T00:00:00+00:00")
    summary = eg.list_run_records(str(tmp_path))[0]
    assert summary["created"] == "2024-06-01T12:00:00+00:00"


def test_an_unpublished_run_still_reports_when_it_was_created(tmp_path):
    """`published_at` is empty until publication, so `created` was blank."""
    _write_run(tmp_path, "rununpub", "2024-06-02T09:30:00+00:00")
    summary = eg.list_run_records(str(tmp_path))[0]
    assert summary["created"] == "2024-06-02T09:30:00+00:00"
    assert summary["published"] == ""


def test_published_time_is_reported_separately(tmp_path):
    _write_run(tmp_path, "runpub", "2024-06-03T00:00:00+00:00",
               published_at="2024-06-04T00:00:00+00:00")
    summary = eg.list_run_records(str(tmp_path))[0]
    assert summary["created"] == "2024-06-03T00:00:00+00:00"
    assert summary["published"] == "2024-06-04T00:00:00+00:00"


# ── the cap keeps the newest, not an arbitrary subset ────────────────── #

def test_the_cap_keeps_the_newest_runs(tmp_path):
    limit = eg.LIST_LIMIT
    total = limit + 5
    for i in range(total):
        # i == 0 is the OLDEST run; give it the alphabetically LAST id, so a
        # filename-sorted-and-then-truncated list would keep exactly the
        # oldest runs.
        rank = total - 1 - i
        run_id = f"r{rank:04d}" + "0" * 8
        created = f"2024-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00"
        _write_run(tmp_path, run_id, created)

    oldest_id = f"r{total - 1:04d}" + "0" * 8
    listed = eg.list_run_records(str(tmp_path))
    assert len(listed) == limit
    assert oldest_id not in _ids(tmp_path), (
        "the cap kept the oldest run, so it was applied before the sort")
    assert listed[0]["run_id"] == "r0000" + "0" * 8, (
        "the newest run is not first")


# ── ordering must not resurrect what the ordering is meant to hide ────── #

def test_quarantined_records_are_still_excluded(tmp_path):
    envelope = {
        "schema_version": eg.ENVELOPE_SCHEMA_VERSION,
        "run_id": "rhist",
        "created_at": "2024-01-01T00:00:00+00:00",
        "record": {"run_id": "rhist", "status": "done",
                   "origin": eg.HISTORICAL_ORIGIN},
        "attestation": {"attested": False, "signature": "missing"},
    }
    (tmp_path / "rhist.json").write_text(json.dumps(envelope),
                                         encoding="utf-8")
    assert _ids(tmp_path) == [], (
        "a quarantined record was listed as a current run")


def test_a_quarantine_stamp_does_not_perturb_the_order(tmp_path):
    """It must be dropped from the ordering, not sorted into it."""
    envelope = {
        "schema_version": eg.ENVELOPE_SCHEMA_VERSION,
        "run_id": "rzzzzzzzzzzz",
        "created_at": "2030-01-01T00:00:00+00:00",   # far future
        "record": {"run_id": "rzzzzzzzzzzz", "status": "done",
                   "origin": eg.HISTORICAL_ORIGIN},
        "attestation": {"attested": False, "signature": "missing"},
    }
    (tmp_path / "rzzzzzzzzzzz.json").write_text(json.dumps(envelope),
                                                encoding="utf-8")
    _write_run(tmp_path, "raaa", "2024-01-01T00:00:00+00:00")
    _write_run(tmp_path, "rbbb", "2024-02-01T00:00:00+00:00")
    assert _ids(tmp_path) == ["rbbb", "raaa"]


def test_unparseable_files_are_skipped_not_fatal(tmp_path):
    _write_run(tmp_path, "rgood", "2024-02-01T00:00:00+00:00")
    (tmp_path / "rbroken.json").write_text("{not json", encoding="utf-8")
    assert _ids(tmp_path) == ["rgood"]


def test_a_record_with_no_created_at_sorts_last(tmp_path):
    """Absence of a time is not a time. It must not masquerade as newest."""
    envelope = {
        "schema_version": eg.ENVELOPE_SCHEMA_VERSION,
        "run_id": "rundated",
        "record": {"run_id": "rundated", "status": "done"},
        "attestation": {"attested": False, "signature": "missing"},
    }
    (tmp_path / "rundated.json").write_text(json.dumps(envelope),
                                            encoding="utf-8")
    _write_run(tmp_path, "rdated", "2020-01-01T00:00:00+00:00")
    assert _ids(tmp_path) == ["rdated", "rundated"], (
        "a run with no created_at was treated as newer than a dated one")


def test_bare_records_without_an_envelope_still_list(tmp_path):
    """Pre-envelope files have no top-level created_at; they must not vanish."""
    (tmp_path / "rlegacy.json").write_text(
        json.dumps({"run_id": "rlegacy", "status": "done",
                    "publication": {"steps_completed": "0/1"}}),
        encoding="utf-8")
    _write_run(tmp_path, "renveloped", "2024-02-01T00:00:00+00:00")
    assert _ids(tmp_path) == ["renveloped", "rlegacy"]
