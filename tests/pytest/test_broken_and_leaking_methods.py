"""Two methods that either crashed or leaked.

Neither was caught by a test, and both are the shape of defect that a passing
suite hides: one could never have returned a value, and one returned the wrong
one every time.

`config_agreement` raised KeyError on every call
------------------------------------------------
It built rows carrying only the field name:

    rows.append({"field": field})

and then concluded with `all(r["agrees"] for r in rows)`. Every `declared` and
`actual` it had just computed was discarded, so the comparison was absent while
looking written. Architecture validation was therefore not merely broken, it was
never running.

A project export leaked every project's reports
-----------------------------------------------
`db.query(ReportRecord).all()`, annotated "For demo, export all or filter by
project". So exporting project A put project B's reports into A's archive. An
export is a zip file the recipient keeps, so that boundary is crossed for good.

Filtering was not available as a patch: `ReportRecord` has no `project_id`
column, so reports are not linked to projects at all and there is nothing to
filter on. The tests below pin the omission and its stated reason, so the leak
cannot return silently, and pin that adding the column is still outstanding.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# ── config_agreement ─────────────────────────────────────────────────────────

class _Spec:
    def __init__(self, **kw):
        self.num_layers = kw.get("num_layers")
        self.num_heads = kw.get("num_heads")
        self.d_model = kw.get("d_model")
        self.model_id = kw.get("model_id", "qwen2.5-1.5b")


class _Config:
    def __init__(self, **kw):
        self.num_hidden_layers = kw.get("num_hidden_layers")
        self.num_attention_heads = kw.get("num_attention_heads")
        self.hidden_size = kw.get("hidden_size")


class _Model:
    def __init__(self, config):
        self.config = config


def _adapter(spec_kwargs, config_kwargs):
    """Build just enough of the adapter for config_agreement to run.

    The method reads `self._model` and `self.spec` and nothing else, so this
    avoids constructing a real model -- which on this repository is known to
    crash the process when five adapter families are built for real.
    """
    from backend.science.models.hf_adapter import HFAdapterMixin

    class _Adapter(HFAdapterMixin):
        def __init__(self):
            self.spec = _Spec(**spec_kwargs)
            self._model = _Model(_Config(**config_kwargs)) if config_kwargs is not None else None

    return _Adapter()


def test_config_agreement_returns_instead_of_raising():
    result = _adapter(
        {"num_layers": 28, "num_heads": 12, "d_model": 1536},
        {"num_hidden_layers": 28, "num_attention_heads": 12, "hidden_size": 1536},
    ).config_agreement()

    assert result["available"] is True
    assert result["spec_matches_config"] is True
    assert result["disagreements"] == []


def test_config_agreement_reports_each_field():
    result = _adapter(
        {"num_layers": 28, "num_heads": 12, "d_model": 1536},
        {"num_hidden_layers": 28, "num_attention_heads": 12, "hidden_size": 1536},
    ).config_agreement()

    fields = {row["field"]: row for row in result["fields"]}
    assert set(fields) == {"num_layers", "num_heads", "d_model"}
    for name, row in fields.items():
        assert "agrees" in row, f"{name} has no 'agrees' key; the caller reads it"
        assert "declared" in row and "actual" in row
        assert row["agrees"] is True


def test_config_agreement_detects_a_mismatch():
    """The reason the method exists: a mis-specified family must not pass.

    d_model 768 is gpt2-small's width. If the spec claims 1536 while the loaded
    config says 768, every measurement would be attributed to the wrong
    architecture.
    """
    result = _adapter(
        {"num_layers": 28, "num_heads": 12, "d_model": 1536},
        {"num_hidden_layers": 28, "num_attention_heads": 12, "hidden_size": 768},
    ).config_agreement()

    assert result["spec_matches_config"] is False
    assert result["disagreements"] == ["d_model"]
    row = next(r for r in result["fields"] if r["field"] == "d_model")
    assert row["declared"] == 1536
    assert row["actual"] == 768
    assert row["agrees"] is False


def test_config_agreement_names_every_disagreeing_field():
    result = _adapter(
        {"num_layers": 12, "num_heads": 12, "d_model": 768},
        {"num_hidden_layers": 28, "num_attention_heads": 12, "hidden_size": 768},
    ).config_agreement()
    assert result["disagreements"] == ["num_layers"]


def test_an_unset_spec_field_is_not_agreement():
    """`None == None` would otherwise read as a match.

    A spec that never declared num_heads and a config that happens to omit the
    attribute both produce None. Comparing them with `==` reports agreement,
    which is exactly the silent pass this method is supposed to prevent.
    """
    result = _adapter(
        {"num_layers": 28, "num_heads": None, "d_model": 1536},
        {"num_hidden_layers": 28, "num_attention_heads": None, "hidden_size": 1536},
    ).config_agreement()

    row = next(r for r in result["fields"] if r["field"] == "num_heads")
    assert row["agrees"] is False, "two absent values were counted as agreeing"
    assert "num_heads" in result["disagreements"]


def test_config_agreement_without_weights_says_so():
    result = _adapter({"num_layers": 28}, None).config_agreement()
    assert result["available"] is False
    assert "no weights loaded" in result["reason"]


# ── Project export scoping ───────────────────────────────────────────────────

def test_reports_cannot_leak_across_projects(tmp_path, monkeypatch):
    """A project's archive must not contain another project's reports."""
    import backend.core.database as database
    from backend.api.project_export import ProjectExporter

    class _Row:
        def __init__(self, session_data):
            self.session_data = session_data

    class _Query:
        def filter(self, *a, **k):
            return self

        def all(self):
            return [_Row({"session_id": "s1", "project": "alpha"})]

    class _Db:
        def query(self, *a, **k):
            return _Query()

        def close(self):
            pass

    # `export_project` imports SessionLocal inside the call, so this is the
    # binding it actually resolves.
    monkeypatch.setattr(database, "SessionLocal", lambda: _Db())
    monkeypatch.setenv("MECH_EXPORT_ROOT", str(tmp_path))

    result = ProjectExporter().export_project("alpha", dest_dir=str(tmp_path))

    with zipfile.ZipFile(result["archive_path"]) as archive:
        names = set(archive.namelist())
        metadata = json.loads(archive.read("metadata.json"))

    assert "reports.json" not in names, (
        "the archive still carries a reports file; reports are not "
        "project-scoped and must not be exported at all")
    assert metadata["reports_included"] == 0
    assert "project_id" in metadata["reports_omitted_reason"], (
        "the omission must state its reason inside the archive, not just drop "
        "the file, so a recipient can tell it apart from an empty project")
    assert metadata["project_id"] == "alpha"
    assert metadata["sessions_included"] == 1
    assert "sessions.json" in names


def test_the_schema_gap_the_real_fix_needs_is_still_open():
    """Documents the outstanding work, and will fail loudly once it is done.

    `ReportRecord.project_id` is the actual fix for project scoping. While the
    column is absent the export omits reports; once someone adds it, this test
    fails and asks them to restore report export with a real filter rather than
    deleting the test.
    """
    from backend.core.database import ReportRecord

    assert not hasattr(ReportRecord, "project_id"), (
        "ReportRecord now has a project_id column, so reports can be filtered "
        "to one project. Restore report export with "
        ".filter(ReportRecord.project_id == project_id) and delete this test.")