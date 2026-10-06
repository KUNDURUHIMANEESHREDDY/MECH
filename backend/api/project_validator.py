"""Project Validator.

Validates that a research project is complete and ready for publication
by checking for required artifacts (models, datasets, provenance, notebooks, etc.).

What this used to return
------------------------
Every check was a literal `True`:

    checks = {
        "model_recorded": True,
        "dataset_recorded": True,
        "checkpoint_hash": True,
        ...
    }
    passed = all(checks.values())

so `validate(project_id)` returned `"ready"` and an empty `missing` list for
every input, including a project id that does not exist. The name of the
function is a claim that something was verified, and nothing was. It was a
rubber stamp on publication readiness, which is the most consequential place in
this codebase to be unable to tell a real check from a constant.

The fix is to make the honest answer the default one. Checks are now produced by
*probes* supplied by the caller, and a check with no probe is `unverified` --
represented as `None`, which is neither a pass nor a failure. `status` is
"ready" only when every check was both probed and passed.

Constructed with no probes, as it is today, this returns "unverified" for every
project rather than "ready". That is a visible regression against the previous
behaviour and it is the intended one: the previous behaviour was the bug. Wiring
a real probe is one line at the call site, and the test suite pins that "ready"
is unreachable without them.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

#: Every check that must pass before a project may be described as ready.
REQUIRED_CHECKS = (
    "model_recorded",
    "dataset_recorded",
    "checkpoint_hash",
    "environment_captured",
    "figures_generated",
    "provenance_tracked",
    "notebook_included",
    "reproducibility_score_calculated",
)

Probe = Callable[[str], bool]


class ProjectValidator:
    """Checks publication readiness, and says what it could not check.

    Args:
        probes: Mapping of check name to a callable taking `project_id` and
            returning a bool. Probes raise rather than return False on an
            infrastructure problem; the exception is caught, recorded, and the
            check is treated as failed, because a check that could not run has
            not passed.
    """

    def __init__(self, probes: Optional[Dict[str, Probe]] = None) -> None:
        self._probes: Dict[str, Probe] = dict(probes or {})

    @property
    def checks(self) -> Iterable[str]:
        return REQUIRED_CHECKS

    def validate(self, project_id: str) -> Dict[str, Any]:
        """Validate publication readiness.

        Returns one of three statuses:

        * ``ready`` -- every required check was probed and passed.
        * ``incomplete`` -- at least one probe ran and failed, so something is
          known to be missing.
        * ``unverified`` -- nothing could be checked. Not a pass, and not a
          claim that anything is missing either.
        """
        checks: Dict[str, Optional[bool]] = {}
        missing: List[str] = []
        unverified: List[str] = []
        errors: Dict[str, str] = {}

        for name in REQUIRED_CHECKS:
            probe = self._probes.get(name)
            if probe is None:
                # `None`, not True and not False. Callers that do
                # `all(checks.values())` get a falsy value, which is the safe
                # direction; callers that read a specific check can tell
                # "unverified" apart from "failed".
                checks[name] = None
                unverified.append(name)
                continue
            try:
                ok = bool(probe(project_id))
            except Exception as exc:  # noqa: BLE001
                # A probe that blew up has not verified anything.
                ok = False
                errors[name] = f"{type(exc).__name__}: {exc}"
            checks[name] = ok
            if not ok:
                missing.append(name)

        if not unverified and not missing:
            status = "ready"
            reason = (f"All {len(checks)} required checks were verified and "
                      f"passed.")
        elif len(unverified) == len(REQUIRED_CHECKS):
            status = "unverified"
            reason = ("No readiness check could be performed, because no probe "
                      "was supplied for any required check. This is not a "
                      "statement that the project is incomplete -- it is a "
                      "statement that nothing was checked.")
        else:
            status = "incomplete"
            if missing:
                reason = (f"{len(missing)} of {len(checks)} checks ran and failed: "
                          + ", ".join(sorted(missing)))
                if unverified:
                    reason += (f". {len(unverified)} further checks could not be "
                               f"run at all: " + ", ".join(sorted(unverified)))
            else:
                reason = (f"{len(checks) - len(unverified)} of {len(checks)} checks were verified, "
                          f"but {len(unverified)} checks could not be run: "
                          + ", ".join(sorted(unverified)))

        result: Dict[str, Any] = {
            "project_id": project_id,
            "status": status,
            "checks": checks,
            "missing": sorted(missing),
            "unverified": sorted(unverified),
            "reason": reason,
        }
        if errors:
            result["check_errors"] = errors
        return result