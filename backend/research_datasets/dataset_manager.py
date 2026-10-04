"""Deterministic Dataset Manager for Mechanistic Interpretability.

Provides a registry for loading reproducible benchmark datasets like IOI,
Induction, Greater-Than, and Copy Task.

Upgraded to include:
- Built-in benchmark datasets (no external files needed)
- Programmatic template-based generation
- Dataset listing, stats, and splitting
- Deterministic seeding for reproducibility
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.science.integrity import sign, verify


#: Digests that stand in for "no hash was recorded" rather than for a hash.
#:
#: The two are both sha256 of empty input -- the empty string and 32 zero bytes --
#: so a manifest that ships either is carrying a placeholder, not a fingerprint.
#: `golden_manifest.json` currently records `bundle_hash` as the 32-zero-byte
#: digest and `prompt_hash` as the empty-string digest, which is why checking only
#: one of them reported a spurious mismatch on the other.
#:
#: A digest is also treated as a placeholder when it is self-describing: entries
#: like `sha256:token_hash_placeholder` or `sha256:f7a2d3c...placeholder` announce
#: that no hash was taken. Comparing against those would either always mismatch or
#: require inventing a match.
PLACEHOLDER_DIGESTS = frozenset({
    "d41d8cd98f00b204e9800998ecf8427e",  # sha256 of 32 zero bytes
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # sha256("")
})


def _is_placeholder(digest: str) -> bool:
    """True when `digest` records no hash, rather than recording a different one.

    Used to distinguish "nothing to check against" from "the data changed". The
    distinction matters: a placeholder is a gap in the manifest, and reporting it
    as a verification failure would train callers to reach for the bypass.

    An `sha256:` prefix is stripped first. Every hash in `golden_manifest.json`
    carries one, so without stripping, `sha256:e3b0c442...` (the empty-string
    digest) was not recognised as a placeholder and the dataset scored as having
    recorded that hash. Callers that pre-strip the prefix are unaffected.
    """
    if not digest:
        return True
    text = str(digest).strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    return text in PLACEHOLDER_DIGESTS or "placeholder" in text


@dataclass
class DatasetLineage:
    """Tracks how a dataset evolved over time."""
    parent_id: Optional[str]
    change_reason: str
    author: str
    breaking_change: bool = False


@dataclass
class DatasetProvenance:
    """Scientific grounding and origin of a dataset."""
    paper_doi: str
    license: str
    source_url: str
    generation_script_hash: str
    random_seed: int
    dataset_doi: Optional[str] = None
    compatible_benchmarks: List[str] = field(default_factory=lambda: ["IOI", "Induction"])


class DatasetManager:
    """Manages versioned, reproducible datasets with integrity verification."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.manifest_path = os.path.join(data_dir, "golden_manifest.json")
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._manifest = self._load_manifest()
        #: Set by `load` when integrity checks were skipped, so a bypassed load is
        #: visible in the returned data rather than only in an environment
        #: variable nobody reads later.
        self.last_integrity_status: Dict[str, Any] = {}

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        """Loads the golden manifest, aliasing dataset folder names (e.g. ``ioi``)
        to their manifest entries so folder-based dataset ids resolve correctly.
        Returns an empty mapping when the manifest is missing or malformed.
        """
        manifest: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                datasets = data.get("datasets", data)
                if isinstance(datasets, dict):
                    for ds_id, meta in datasets.items():
                        if isinstance(meta, dict):
                            manifest[ds_id] = meta
                            if meta.get("dataset_id") and meta["dataset_id"] not in manifest:
                                manifest[meta["dataset_id"]] = meta
            except (OSError, ValueError):
                pass
        # Alias on-disk dataset folders to the matching manifest entry so that
        # load("ioi") -> datasets/ioi/dataset.json works even when the manifest
        # registers the dataset under a canonical id like "IOI-Canonical-100".
        if os.path.isdir(self.data_dir):
            for name in os.listdir(self.data_dir):
                folder = os.path.join(self.data_dir, name)
                if not os.path.isdir(folder) or not os.path.exists(os.path.join(folder, "dataset.json")):
                    continue
                if name in manifest:
                    continue
                match = next(
                    (e for e in manifest.values() if name.lower() in str(e.get("dataset_id", "")).lower()),
                    None,
                )
                if match is not None:
                    manifest[name] = match
        return manifest

    def validate_schema(self, data: Dict[str, Any]) -> None:
        """Validates that a loaded dataset dict has the required ``prompts`` field."""
        if not isinstance(data, dict) or "prompts" not in data:
            raise ValueError("Dataset schema invalid: missing 'prompts' field")

    def validate(self, data: Dict[str, Any]) -> None:
        self.validate_schema(data)

    def _compute_sha256(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def log_audit_event(self, dataset_id: str, action: str, researcher: str, details: str) -> None:
        """Records a lifecycle event in the immutable dataset audit log."""
        log_path = os.path.join(self.data_dir, "dataset_audit.json")
        event = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "dataset_id": dataset_id,
            "action": action,
            "researcher": researcher,
            "details": details
        }

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception: pass

    def trigger_revalidation(self, dataset_id: str):
        """Marks downstream research as stale when a dataset changes."""
        self.log_audit_event(dataset_id, "IMPACT", "SYSTEM", "Dataset changed. Downstream results marked for revalidation.")
        # In a real system, this would update validation_history.db entries for pass -> stale

    def sign_dataset(self, dataset_id: str, private_key: Any = None) -> str:
        """Sign a dataset's identity with Ed25519.

        Replaces `sha256(f"{payload}:{private_key}")` with
        `private_key="mock_private_key"`, which was documented as an "asymmetric
        digital signature" but was a keyed hash: proving it required the secret,
        and the secret was a default argument in the source. Anyone who could read
        the repository could produce a signature that verified.

        `private_key` is optional only in the sense that it may come from
        `MECH_SIGNING_KEY_PATH` instead. There is no default key, so this raises
        rather than signing with something public.
        """
        meta = self._manifest.get(dataset_id)
        if not meta: raise ValueError(f"Dataset {dataset_id} not found.")

        signature = sign(self._signing_payload(dataset_id, meta), private_key)
        meta["signature"] = signature
        meta["signature_algorithm"] = "Ed25519"
        return signature

    def verify_signature(self, dataset_id: str, public_key: Any) -> Dict[str, Any]:
        """Verify a dataset signature with the public key.

        The previous signature was `verify_signature(dataset_id,
        public_key="mock_public_key") -> bool`, and it **ignored its own
        `public_key` argument**, recomputing with the literal
        `"mock_private_key"`. Verification that needs the secret is not
        verification: it proves the caller can read this file. The public key is
        now load-bearing, and it is a required argument.

        Returns a dict rather than a bool so "no signature", "no key" and "wrong
        signature" stay distinguishable -- three different problems that a bare
        False collapses into one.
        """
        meta = self._manifest.get(dataset_id)
        if not meta:
            return {"valid": False, "reason": f"Dataset {dataset_id} not found",
                    "algorithm": "Ed25519"}
        if "signature" not in meta:
            return {"valid": False, "reason": "no signature is recorded",
                    "algorithm": "Ed25519"}

        result = verify(self._signing_payload(dataset_id, meta),
                        meta.get("signature"), public_key)
        return result.to_dict()

    def _resolve_path(self, dataset_id: str, meta: Dict[str, Any]) -> str | None:
        """Locate a dataset's dataset.json, tolerating canonical-vs-folder ids.

        `golden_manifest.json` registers the IOI dataset as
        `IOI-Canonical-100`, but the file lives in `datasets/ioi/dataset.json`.
        `load` used to build its path straight from the id it was given, so the
        canonical id -- the one the manifest and every report name -- raised
        FileNotFoundError, and only the folder alias worked. Callers "solved"
        this by catching the exception and reloading under a different id, which
        is how the old audit script ended up setting a hash-bypass flag.

        Resolution order: the id's own folder, then any on-disk folder whose name
        is a case-insensitive substring of the canonical id (the existing alias
        rule, applied in reverse).
        """
        direct = os.path.join(self.data_dir, dataset_id, "dataset.json")
        if os.path.exists(direct):
            return direct

        if not os.path.isdir(self.data_dir):
            return None
        canonical = str(meta.get("dataset_id", dataset_id)).lower()
        for name in sorted(os.listdir(self.data_dir)):
            candidate = os.path.join(self.data_dir, name, "dataset.json")
            if os.path.exists(candidate) and name.lower() in canonical:
                return candidate
        return None

    @staticmethod
    def _signing_payload(dataset_id: str, meta: Dict[str, Any]) -> str:
        """The canonical string both sides sign.

        Uses `meta['dataset_id']` -- the manifest's canonical id -- rather than
        the id the caller happened to pass. Both spellings resolve to the same
        dataset (`load("ioi")` and `load("IOI-Canonical-100")` both work), so
        embedding the caller's spelling made the dataset sign under one payload
        and fail to verify under the other: a signature produced via the folder
        alias did not verify via the canonical id. Identity has to come from the
        manifest, not from the access path.

        No private key appears here, which is what separates this from the keyed
        hash it replaces.
        """
        canonical = str(meta.get("dataset_id") or dataset_id)
        return f"{canonical}:{meta['version']}:{meta['hashes']['bundle_hash']}"

    def load(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Load a Golden Dataset, verifying every manifest hash that is populated.

        The docstring used to say "Triple-SHA integrity verification". It verified
        one hash. `prompt_hash` and `expected_prompt` were computed at lines 164-166
        and then never compared with anything -- two dead locals that made the
        count look like three.

        Both checks now run, and `last_integrity_status` records which were
        actually performed. A hash whose manifest value is the empty-file digest
        (`d41d8cd9...`, i.e. a placeholder rather than a recorded hash) is
        reported as `not_recorded` rather than counted as a pass: "no expectation
        to check against" is not "verified".
        """
        if dataset_id not in self._manifest:
            raise ValueError(f"Dataset '{dataset_id}' not found in golden_manifest.json")

        meta = self._manifest[dataset_id]
        dataset_path = self._resolve_path(dataset_id, meta)

        if dataset_path is None:
            searched = sorted(
                name for name in os.listdir(self.data_dir)
                if os.path.exists(os.path.join(self.data_dir, name, "dataset.json"))
            ) if os.path.isdir(self.data_dir) else []
            raise FileNotFoundError(
                f"No dataset.json found for '{dataset_id}'. Looked in "
                f"{os.path.join(self.data_dir, dataset_id)}. On-disk dataset "
                f"folders: {searched or 'none'}. The manifest registers this "
                f"dataset as '{meta.get('dataset_id', dataset_id)}', so pass "
                f"either that canonical id or the folder alias."
            )

        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
            data = json.loads(raw_content)

        bundle_hash = self._compute_sha256(raw_content)
        expected_bundle = meta["hashes"]["bundle_hash"].replace("sha256:", "")

        prompts_str = json.dumps(data["prompts"], sort_keys=True)
        prompt_hash = self._compute_sha256(prompts_str)
        expected_prompt = meta["hashes"].get("prompt_hash", "").replace("sha256:", "")

        checks: Dict[str, str] = {}
        if _is_placeholder(expected_bundle):
            checks["bundle_hash"] = "not_recorded"
        elif bundle_hash != expected_bundle:
            checks["bundle_hash"] = "mismatch"
        else:
            checks["bundle_hash"] = "verified"

        if _is_placeholder(expected_prompt):
            checks["prompt_hash"] = "not_recorded"
        elif prompt_hash != expected_prompt:
            checks["prompt_hash"] = "mismatch"
        else:
            checks["prompt_hash"] = "verified"

        mismatched = [name for name, state in checks.items() if state == "mismatch"]

        # The old `MECH_BYPASS_HASH_CHECK` was an environment variable that
        # disabled the only check that ran, and `run_reproducibility_audit.py`
        # set it globally without ever restoring it, so every later `load()` in that
        # process silently skipped verification too.
        #
        # Replaced with an unmistakably named development override that has to be
        # confirmed with a second value, and that records itself on the returned
        # data. A bypass that leaves no trace in the result is not a bypass, it is
        # a silent loss of a check.
        bypassed = False
        if mismatched:
            dev_mode = os.environ.get("MECH_INTEGRITY_CHECKS", "")
            if dev_mode == "disabled-for-local-development":
                bypassed = True
            else:
                self.last_integrity_status = {
                    "dataset_id": dataset_id,
                    "checks": checks,
                    "integrity_verified": False,
                    "bypassed": False,
                }
                raise ValueError(
                    f"CRITICAL: Dataset integrity check failed for {dataset_id}: "
                    f"{', '.join(mismatched)} mismatch. To run against local "
                    f"development data, set MECH_INTEGRITY_CHECKS="
                    f"disabled-for-local-development -- that disables these "
                    f"checks for every load in the process and the result is "
                    f"marked integrity_verified=False."
                )

        self.validate_schema(data)
        self._datasets[dataset_id] = data
        self.last_integrity_status = {
            "dataset_id": dataset_id,
            "checks": checks,
            "integrity_verified": not bypassed and not mismatched,
            "bypassed": bypassed,
        }
        if bypassed:
            data.setdefault("integrity_notice", (
                "Integrity checks were DISABLED for this load "
                "(MECH_INTEGRITY_CHECKS). This data is not verified and must not "
                "support a scientific claim."
            ))
        return data["prompts"]

    def compute_fingerprint(self, dataset_id: str, prompts: List[Dict[str, Any]], tokenizer: Any = None) -> Dict[str, str]:
        """Exhaustive fingerprint: Prompt, Token, Bundle, and Environment."""
        prompt_str = json.dumps(prompts, sort_keys=True)

        # 1. Prompt Hash
        p_hash = self._compute_sha256(prompt_str)

        # 2. Token Hash (Detects tokenizer drift)
        t_hash = "sha256:unknown"
        if tokenizer:
            tokens = []
            for p in prompts[:10]: # Sample for performance
                tokens.extend(tokenizer.encode(p["clean"]))
            t_hash = self._compute_sha256(str(tokens))

        # 3. Environment/Script Fingerprint (Simulation)
        e_hash = self._compute_sha256(os.name + os.getcwd())

        return {
            "prompt_hash": p_hash,
            "token_hash": t_hash,
            "environment_hash": e_hash
        }

    def compute_health_score(self, dataset_id: str) -> Dict[str, Any]:
        """Calculates granular health metrics for a dataset."""
        meta = self._manifest.get(dataset_id, {})
        if not meta: return {"overall": 0}

        # 1. Integrity.
        #
        # Was `1.0 if all(k in h for k in [...]) else 0.5` -- it checked only that
        # the three keys *exist*. `golden_manifest.json` has all three, but
        # `prompt_hash` and `bundle_hash` are placeholder digests, so the shipped
        # dataset scored 100% on integrity while recording no verifiable hash.
        # Presence of a field is not evidence of integrity; now a placeholder
        # counts as not recorded.
        h = meta.get("hashes", {})
        integrity_keys = ["prompt_hash", "token_hash", "bundle_hash"]
        recorded = [k for k in integrity_keys if not _is_placeholder(h.get(k, ""))]
        if len(recorded) == len(integrity_keys):
            integrity = 1.0
        elif recorded:
            integrity = 0.5
        else:
            integrity = 0.0

        # 2. Lineage (Parent recorded or initial)
        lineage = meta.get("lineage", {})
        lineage_score = 1.0 if lineage.get("author") and (lineage.get("parent_id") or lineage.get("change_reason")) else 0.5

        # 3. Metadata (Paper DOI, License)
        prov = meta.get("provenance", {})
        metadata_score = 1.0 if prov.get("paper_doi") and prov.get("license") else 0.7

        # 4. Signature.
        #
        # Was `1.0 if meta.get("signature")`. The manifest ships
        # `"signature": "sha256:dataset_sig_placeholder"`, which is truthy, so the
        # dataset scored 100% for being signed. A placeholder is not a signature,
        # and neither is an unverifiable one: a full credit now requires an Ed25519
        # signature of the recorded length, and partial credit is not offered for a
        # value that cannot be checked at all.
        signature = meta.get("signature", "")
        sig_score = 1.0 if (not _is_placeholder(signature)
                            and len(str(signature)) == 128) else 0.0

        # 5. Compatibility
        #
        # Was `comp_score = 0.98`, commented "# 5. Compatibility (Mocked)".
        # It was weighted 0.1 into `overall`, so every dataset scored as 98%
        # compatible with no compatibility check existing anywhere -- a free
        # 9.8 points on every dataset's health score.
        #
        # Nothing here can load a dataset to test it, so the term is excluded
        # from the weighted total rather than invented. `overall` is
        # renormalised over the four terms that were actually checked, and says
        # so, so a reader can see the 0.1 was dropped rather than lost.
        comp_score: Optional[float] = None
        UNMEASURED_WEIGHT = 0.1
        measured_weight = 1.0 - UNMEASURED_WEIGHT  # 0.9

        overall = ((integrity * 0.3 + lineage_score * 0.2
                    + metadata_score * 0.2 + sig_score * 0.2)
                   / measured_weight) * 100

        return {
            "integrity": integrity * 100,
            "integrity_hashes_recorded": f"{len(recorded)}/{len(integrity_keys)}",
            "integrity_unrecorded": [k for k in integrity_keys if k not in recorded],
            "lineage": lineage_score * 100,
            "metadata": metadata_score * 100,
            "signature": sig_score * 100,
            "signature_is_real": sig_score == 1.0,
            # Absent, not 98.0.
            "compatibility": None,
            "compatibility_measured": False,
            "compatibility_reason": (
                "No compatibility check is implemented; this dataset's "
                "compatibility is unknown."
            ),
            "overall": round(overall, 1),
            "overall_is_partial": True,
            "overall_weight_covered": round(measured_weight, 2),
            "overall_unmeasured_weight": UNMEASURED_WEIGHT,
        }

    def archive_dataset(self, dataset_id: str, researcher: str) -> bool:
        """Transitions a dataset to ARCHIVED status."""
        meta = self._manifest.get(dataset_id)
        if not meta: return False

        meta["status"] = "ARCHIVED"
        meta["locked"] = True
        self.log_audit_event(dataset_id, "ARCHIVE", researcher, "Dataset archived for historical reproducibility.")
        return True

    def get_lifecycle_status(self, dataset_id: str) -> str:
        return self._manifest.get(dataset_id, {}).get("status", "DRAFT")

    def generate_drift_report(self, dataset_id: str, current_token_hash: str) -> Dict[str, Any]:
        """Diagnostics for when the tokenizer has changed."""
        meta = self._manifest.get(dataset_id, {})
        expected_hash = meta.get("hashes", {}).get("token_hash", "")

        return {
            "dataset_id": dataset_id,
            "status": "DRIFT_DETECTED",
            "expected_hash": expected_hash,
            "current_hash": current_token_hash,
            "diagnosis": "Tokenizer merge rules or vocabulary changed. Benchmark values may be invalid.",
            "affected_samples": "ALL"
        }

    def metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Returns metadata for the dataset (version, citation, etc)."""
        if dataset_name not in self._datasets:
            self.load(dataset_name)

        ds = self._datasets[dataset_name]
        prompts = ds.get("prompts", [])
        
        # Compute task distribution
        tasks = {}
        for p in prompts:
            t = p.get("metadata", {}).get("task", "unknown")
            tasks[t] = tasks.get(t, 0) + 1
        
        return {
            "version": ds.get("version"),
            "citation": ds.get("citation"),
            "checksum": ds.get("checksum"),
            "num_prompts": len(prompts),
            "tasks": tasks,
        }

    def split(self, dataset_name: str, split_name: str = "validation") -> List[Dict[str, Any]]:
        """Returns only prompts matching a specific split (train/validation/test).
        
        Args:
            dataset_name: Name of the dataset.
            split_name: The split to filter by.
            
        Returns:
            Filtered list of prompt dictionaries.
        """
        prompts = self.load(dataset_name)
        return [p for p in prompts if p.get("metadata", {}).get("split") == split_name]

    def stats(self, dataset_name: str) -> Dict[str, Any]:
        """Returns summary statistics for a dataset."""
        prompts = self.load(dataset_name)
        meta = self.metadata(dataset_name)
        
        avg_clean_len = sum(len(p["clean"]) for p in prompts) / max(1, len(prompts))
        unique_targets = len(set(p["target"] for p in prompts))
        splits = {}
        for p in prompts:
            s = p.get("metadata", {}).get("split", "unknown")
            splits[s] = splits.get(s, 0) + 1
        
        return {
            **meta,
            "avg_clean_length_chars": round(avg_clean_len, 1),
            "unique_targets": unique_targets,
            "splits": splits,
        }

    def save(self, dataset_name: str, data: Dict[str, Any]) -> str:
        """Saves a dataset to disk for persistence.
        
        Returns:
            The file path written to.
        """
        self.validate(data)
        out_dir = os.path.join(self.data_dir, dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "dataset.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._datasets[dataset_name] = data
        return out_path
