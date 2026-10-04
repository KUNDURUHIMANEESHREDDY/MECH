"""Automated Hypothesis Generator Engine.

Formats measured findings into testable, falsifiable hypothesis records.

What this used to be
--------------------
A function that ignored its argument and returned two literals::

    generate_hypotheses(context_prompt="The capital of France is")

    -> [
        {"hypothesis_id": "hyp_gen_1",
         "statement": "Attention Head L8_H9 acts as an induction head routing "
                      "geographic entity tokens.",
         "evidence": ["L8_H9 attention weight 0.85 on 'France'",
                      "Correlation r=0.91 with capital probe"],
         "confidence": 0.89,
         "suggested_experiment": "Causal tracing & head ablation on L8_H9 "
                                "over city/country pairs"},
        {"hypothesis_id": "hyp_gen_2",
         "statement": "SAE Feature #1402 is polysemantic for both IOI names "
                      "and comma syntax.",
         "evidence": ["High activation on 'Mary'",
                      "Moderate activation on comma tokens"],
         "confidence": 0.65,
         "suggested_experiment": "Polysemanticity detection & co-firing "
                                "clustering over OpenWebText"},
    ]

Nothing there was derived from anything. Specifically:

  * `context_prompt` was never read. The capital-of-France prompt had no
    connection to either hypothesis, so the function's only input was decorative.
  * Both evidence lists were invented measurements -- an attention weight of 0.85,
    a correlation of r=0.91, activations on the tokens "Mary" and on commas. No
    run produced them.
  * Hypothesis 1 called L8_H9 an *induction* head. L8_H9 is this project's IOI
    name-mover head; the induction-heads pipeline is a separate measurement that
    reports its own heads. The statement also contradicted hypothesis 2, which
    places IOI names on an SAE feature.
  * Hypothesis 2 concerned "SAE Feature #1402". There is no SAE in this
    repository -- see `feature_auto_interpreter`, whose former activation function
    was a substring matcher -- so there is no feature 1402.
  * "over OpenWebText" named a dataset that is not present.
  * The confidences 0.89 and 0.65 described nothing.

What it does now
----------------
Findings come from the caller. Each carries the measurement it rests on and the
provenance of that measurement, and the record says which of them were actually
observed. A finding whose provenance is not a measurement is retained but marked
`hypothesis_status: "speculative"`, so it can still be proposed -- an unmeasured
idea is a legitimate thing to propose, provided nothing claims otherwise.

With no findings, nothing is generated. The generator does not invent subjects to
have opinions about.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Any, Dict, List, Optional, Sequence

#: Provenance values that mean "this was actually measured". Anything else leaves
#: the hypothesis marked speculative.
_MEASURED = {"live", "measured", "observed"}


class HypothesisGeneratorEngine:
    """Formats supplied findings into hypothesis records.

    Proposes nothing on its own. It cannot: it has no model, no data, and no way
    to evaluate a claim.
    """

    def __init__(self, provenance_reader: Optional[Any] = None) -> None:
        #: Optional callable mapping a finding dict to a provenance label, so a
        #: caller can point at its own convention (this codebase reads
        #: `provenance_of`, which checks source/kind/type/status).
        self._provenance_reader = provenance_reader

    def generate_hypotheses(
        self,
        context_prompt: Optional[str] = None,
        findings: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Turn measured findings into hypothesis records.

        Each finding is ``{"statement", "evidence", "suggested_experiment",
        "provenance"}``. `evidence` entries are ``{"claim", "value"}`` and must
        carry a numeric `value` -- a string like "high activation" cannot be
        checked by a reader and is not accepted.

        Returns an empty list when nothing was supplied, together with a
        `self.last_generation_note` explaining why. It does not fall back to
        invented hypotheses.
        """
        if not findings:
            self.last_generation_note = (
                "No findings were supplied, so no hypotheses were generated. "
                "This previously returned two fixed hypotheses about L8_H9 and "
                "SAE feature #1402, with invented evidence and confidences, "
                "regardless of the context prompt."
            )
            return []

        records: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        for index, finding in enumerate(findings, start=1):
            if not isinstance(finding, dict):
                skipped.append({"index": index, "reason": "not a dict"})
                continue

            statement = finding.get("statement")
            if not isinstance(statement, str) or not statement.strip():
                skipped.append({"index": index, "reason": "no statement"})
                continue

            evidence, evidence_skipped = self._normalise_evidence(
                finding.get("evidence"))
            if evidence_skipped:
                skipped.append({"index": index, "reason": evidence_skipped})

            provenance = self._provenance_of(finding)
            measured = provenance in _MEASURED

            records.append({
                "hypothesis_id": f"hyp_gen_{uuid.uuid4().hex[:8]}",
                "statement": statement.strip(),
                "evidence": evidence,
                "n_evidence_items": len(evidence),
                "provenance": provenance,
                # An unmeasured idea may still be worth proposing, but it must not
                # arrive wearing a confidence.
                "confidence": None,
                "confidence_reason": (
                    "Derived from the supplied evidence."
                    if measured and evidence else
                    "Not established: this hypothesis rests on no measured "
                    "evidence, and this engine cannot evaluate one."
                ),
                "hypothesis_status": (
                    "evidence_backed" if measured and evidence
                    else "speculative"),
                "verified": False,
                "suggested_experiment": finding.get("suggested_experiment"),
                "context_prompt": context_prompt,
                "generated_at": _dt.datetime.now(
                    _dt.timezone.utc).isoformat(),
            })

        self.last_generation_note = (
            f"{len(records)} hypothesis record(s) from {len(findings)} finding(s)"
            + (f"; {len(skipped)} skipped" if skipped else "")
        )
        return records

    last_generation_note: str = ""

    def _normalise_evidence(
        self,
        evidence: Any,
    ) -> tuple:
        """Keep only evidence items carrying a checkable numeric value.

        A claim like "High activation on 'Mary'" is not evidence a reader can
        assess. An item needs a `value` that is a number, and a `source` saying
        where that number came from.
        """
        if not isinstance(evidence, (list, tuple)):
            return [], "evidence is not a list"

        kept: List[Dict[str, Any]] = []
        dropped = 0
        for item in evidence:
            if not isinstance(item, dict):
                dropped += 1
                continue
            value = item.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                dropped += 1
                continue
            kept.append({
                "claim": item.get("claim"),
                "value": value,
                "source": item.get("source"),
                "measured": item.get("measured"),
            })
        return kept, (f"{dropped} evidence item(s) lacked a numeric value"
                      if dropped else None)

    def _provenance_of(self, finding: Dict[str, Any]) -> Optional[str]:
        """Read the finding's provenance, falling back to the injected reader."""
        if self._provenance_reader is not None:
            try:
                label = self._provenance_reader(finding)
                if label:
                    return str(label)
            except Exception:
                # A reader that cannot cope with this record should not make the
                # finding look measured; it falls through to the explicit field.
                pass
        label = finding.get("provenance")
        if isinstance(label, str):
            return label
        source = finding.get("source")
        return source if isinstance(source, str) else None