"""A validator that could only ever agree.

`ProjectValidator.validate` checked eight conditions and every one of them was a
literal `True`. `all(checks.values())` was therefore always true, so every
project -- including one that does not exist -- came back `"ready"` with an empty
`missing` list.

The dangerous part is not that it was wrong. It is that it was wrong in the
direction that authorises publication, and that its name and its return shape
were indistinguishable from a real check. A caller cannot tell a verified
project from a stub, which is the whole problem this repository exists to
address, sitting on the one function whose output means "you may publish this".

These tests pin the fail-closed behaviour: no probe means unverified, and
"ready" is unreachable until every required check is both probed and passed.
"""
from __future__ import annotations

import pytest


def test_no_probe_means_not_ready():
    from backend.api.project_validator import ProjectValidator

    result = ProjectValidator().validate("any-project")
    assert result["status"] != "ready"
    assert result["status"] == "unverified"
    assert result["missing"] == []
    assert sorted(result["unverified"]) == sorted(ProjectValidator().checks)


def test_an_unknown_project_is_not_ready():
    """The case the old version called "ready" most confidently."""
    from backend.api.project_validator import ProjectValidator

    result = ProjectValidator().validate("project-that-does-not-exist")
    assert result["status"] != "ready"


def test_every_check_is_present_and_unverified_by_default():
    from backend.api.project_validator import REQUIRED_CHECKS, ProjectValidator

    result = ProjectValidator().validate("p")
    assert set(result["checks"]) == set(REQUIRED_CHECKS)
    assert len(REQUIRED_CHECKS) == 8, (
        "a required check went missing; readiness must not get easier by "
        "dropping entries from the list")
    for name, value in result["checks"].items():
        assert value is None, f"{name} is {value!r}, not None"


def test_all_checks_passing_is_required_for_ready():
    from backend.api.project_validator import REQUIRED_CHECKS, ProjectValidator

    probes = {name: (lambda _pid: True) for name in REQUIRED_CHECKS}
    result = ProjectValidator(probes).validate("p")
    assert result["status"] == "ready"
    assert result["missing"] == []
    assert result["unverified"] == []


def test_one_failing_check_is_enough_to_block():
    from backend.api.project_validator import REQUIRED_CHECKS, ProjectValidator

    probes = {name: (lambda _pid: True) for name in REQUIRED_CHECKS}
    probes["checkpoint_hash"] = lambda _pid: False
    result = ProjectValidator(probes).validate("p")

    assert result["status"] == "incomplete"
    assert result["missing"] == ["checkpoint_hash"]
    assert result["checks"]["checkpoint_hash"] is False


def test_a_partial_set_of_probes_cannot_report_ready():
    """Seven verified checks are not eight checks."""
    from backend.api.project_validator import REQUIRED_CHECKS, ProjectValidator

    probes = {name: (lambda _pid: True)
              for name in REQUIRED_CHECKS if name != "notebook_included"}
    result = ProjectValidator(probes).validate("p")

    assert result["status"] == "incomplete"
    assert result["unverified"] == ["notebook_included"]
    assert "notebook_included" in result["reason"]


def test_a_probe_that_raises_has_not_passed():
    """A check that could not run is not a passing check."""
    from backend.api.project_validator import REQUIRED_CHECKS, ProjectValidator

    def _explode(_pid):
        raise RuntimeError("registry unreachable")

    probes = {name: (lambda _pid: True) for name in REQUIRED_CHECKS}
    probes["provenance_tracked"] = _explode
    result = ProjectValidator(probes).validate("p")

    assert result["status"] == "incomplete"
    assert result["checks"]["provenance_tracked"] is False
    assert "provenance_tracked" in result["check_errors"]
    assert "registry unreachable" in result["check_errors"]["provenance_tracked"]


def test_the_probes_receive_the_project_id():
    from backend.api.project_validator import REQUIRED_CHECKS, ProjectValidator

    seen = []

    def _record(pid):
        seen.append(pid)
        return True

    probes = {name: _record for name in REQUIRED_CHECKS}
    ProjectValidator(probes).validate("the-project")
    assert seen and set(seen) == {"the-project"}


def test_all_checks_values_is_safe_on_an_unverified_result():
    """Callers doing `all(checks.values())` must get False, not a silent True.

    This is the specific shape of the old bug: a dict of booleans that happened
    to be all-True. `None` is falsy, so the naive check fails closed.
    """
    from backend.api.project_validator import ProjectValidator

    checks = ProjectValidator().validate("p")["checks"]
    assert all(checks.values()) is False