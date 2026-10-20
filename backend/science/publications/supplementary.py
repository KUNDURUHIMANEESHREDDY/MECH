"""Packaging for research artifacts, with a label the data can support.

The supplementary bundle summarises an experiment somebody else ran, so the
only thing this module may do with provenance is *derive* it. It used to
assert ``{"measured": True, "provenance": "live"}`` over whatever traces it
was handed -- a reference run, a seeded fixture and an empty list all came
out labelled live -- and an empty list came out as a live measurement of
zero samples.

The rule
--------

The summary takes the weakest label its inputs carry, and a label is live
only when something attested it:

* no trace list at all -> ``unavailable``: nothing was supplied to count.
* an empty trace list -> ``unavailable``: zero samples is not a measurement.
* every trace labelled live -> ``live``.
* any other label found -> that weakest label (``seeded``, ``reference`` or
  ``unavailable``), with a reason naming the labels that were found. A
  summary of seeded data is seeded; it is not upgraded to live.
* nothing labelled at all -> ``unavailable``: the origin cannot be
  established, so the count is not reported as a measurement.

Traces usually carry no provenance of their own -- a pipeline envelope
attests its ``raw_traces`` as part of the payload it attests -- so when the
traces say nothing, the label of the envelope that carries them is used.
That label was issued by ``attest_measurement``, so it is inherited rather
than invented.
"""

import os
import json
import shutil
from typing import Any, Dict, List, Optional, Set

from backend.core.provenance import LIVE, REFERENCE, SEEDED, UNAVAILABLE

#: Provenance labels, weakest first. A summary takes the weakest label any of
#: its inputs carries, so a summary of seeded data is seeded, and a summary
#: mixing a live trace with a reference trace is a reference.
_LABEL_ORDER = (UNAVAILABLE, REFERENCE, SEEDED, LIVE)


class SupplementaryMaterialPackager:
    """
    Packages all research artifacts into a comprehensive supplementary material bundle.
    """

    def __init__(self, experiment_id: str, results: Dict[str, Any],
                 output_root: str = "backend/science/publications/packages"):
        self.experiment_id = experiment_id
        self.results = results if isinstance(results, dict) else {}
        self.package_path = os.path.join(output_root, f"{experiment_id}_supplementary")
        os.makedirs(self.package_path, exist_ok=True)

    def _traces(self) -> Optional[List[Any]]:
        """The raw samples this result was given, or None if it carries none."""
        for key in ("raw_traces", "traces", "samples"):
            candidate = self.results.get(key)
            if isinstance(candidate, list):
                return candidate
        return None

    @staticmethod
    def _label_of(item: Any) -> Optional[str]:
        """The provenance an item declares about itself, if it declares one."""
        if isinstance(item, dict):
            label = item.get("provenance")
            if isinstance(label, str) and label:
                return label
        return None

    @staticmethod
    def _weakest(labels: Set[str]) -> str:
        """The weakest label in the set, by the order above.

        A label nobody can interpret is treated as no stronger than
        ``reference``: it must not be read as a measurement.
        """
        for candidate in _LABEL_ORDER:
            if candidate in labels:
                return candidate
        return REFERENCE

    def _raw_data_summary(self) -> Dict[str, Any]:
        """Describe the raw data that was actually supplied, and label it honestly.

        The label is derived from the data, never asserted. See the module
        docstring for the rule; in short, `live` survives only when the
        traces that were supplied say they are live.
        """
        traces = self._traces()
        if traces is None:
            return {
                "measured": False,
                "provenance": UNAVAILABLE,
                "reason": ("No raw samples were supplied with this result, so no "
                           "sample count or metric is reported."),
            }

        if not traces:
            return {
                "measured": False,
                "provenance": UNAVAILABLE,
                "reason": ("The supplied trace list was empty, so there are no "
                           "samples to count and no measurement to report."),
            }

        labels = {label for label in (self._label_of(t) for t in traces) if label}
        if not labels:
            # The traces say nothing about themselves. The envelope that
            # carries them may: a pipeline envelope is attested as a whole,
            # raw_traces included, so its label is the label of its data.
            envelope = self._label_of(self.results)
            labels = {envelope} if envelope else set()

        if not labels:
            return {
                "measured": False,
                "provenance": UNAVAILABLE,
                "reason": ("The supplied traces carry no provenance of their own "
                           "and the envelope that carries them is unlabelled, so "
                           "their origin cannot be established and no sample "
                           "count is reported as measured."),
            }

        label = self._weakest(labels)
        if label == UNAVAILABLE:
            return {
                "measured": False,
                "provenance": UNAVAILABLE,
                "reason": (f"The supplied traces are labelled "
                           f"{', '.join(sorted(labels))}, so they are not a "
                           "measurement and no sample count is reported."),
            }

        metrics = sorted({
            str(t.get("metric"))
            for t in traces
            if isinstance(t, dict) and t.get("metric")
        })
        return {
            "measured": True,
            "provenance": label,
            "samples": len(traces),
            "metrics": metrics,
            "reason": ("" if label == LIVE else
                       f"The summary counts the supplied traces, which are "
                       f"labelled {', '.join(sorted(labels))}; that is not a "
                       "live measurement."),
        }

    def create_bundle(self):
        """
        Gathers and organizes all required artifacts.
        """
        # 1. Metadata and Reports
        self._save_json("research_manifest.json", self.results.get("manifest", {}))
        self._save_json("statistical_report.json", self.results.get("statistical_results", {}))
        self._save_json("benchmark_certificate.json", self.results.get("certificate", {}))
        self._save_json("environment_snapshot.json", self.results.get("hardware_passport", {}))

        # 2. Figures and Tables
        fig_src = os.path.join("backend/science/publications/figures", self.experiment_id)
        if os.path.exists(fig_src):
            shutil.copytree(fig_src, os.path.join(self.package_path, "figures"),
                            dirs_exist_ok=True)

        # 3. Raw Data provenance (counted, or explicitly absent)
        os.makedirs(os.path.join(self.package_path, "raw_data"), exist_ok=True)
        self._save_json("raw_data/activation_samples_summary.json",
                        self._raw_data_summary())

        return self.package_path

    def _save_json(self, filename: str, data: Any):
        path = os.path.join(self.package_path, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
