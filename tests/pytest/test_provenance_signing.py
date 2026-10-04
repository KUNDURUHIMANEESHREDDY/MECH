"""Guards on provenance: signing must be real, and a run must not assert what
it did not measure.

Each test here corresponds to a specific defect that was present:

* ``sha256(payload + ":" + "mock_private_key")`` described as an "asymmetric
  digital signature", and verified by recomputing with that same literal --
  so anyone who could read the repository could forge a signature that verified.
* ``ResearchManifestEngine.sign_manifest(private_key="mock_key")`` returning a
  keyed SHA-256 under a docstring reading "Simulating Ed25519".
* ``MECH_BYPASS_HASH_CHECK``, an environment variable that disabled the only
  hash check that ran, set globally by a script and never restored.
* ``DatasetManager.load`` computing ``prompt_hash`` and ``expected_prompt`` and
  then never comparing them, under a docstring claiming "Triple-SHA".
* ``validate_benchmark(patch_success_rate=100.0)`` -- an unmeasured check that
  defaulted to passing.
* ``ResearchManifest.status = "VALIDATED"`` as a literal, whatever the verdict.
* A "reproducibility audit" that generated its observations with
  ``[0.88 + (0.02 * (0.5 - i/100.0)) for i in range(100)]``.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.pytest.source_assert import doc_and_comments, executable_source

from backend.science.integrity import SigningUnavailable, sign, verify
from backend.science.reproducibility.research_manifest import (
    ResearchManifestEngine,
    derive_manifest_status,
)
from backend.science.reproducibility.scientific_validator import ScientificValidator
from backend.research_datasets.dataset_manager import (
    DatasetManager,
    _is_placeholder,
)


# ── Real signing ───────────────────────────────────────────────────────────

def _keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    seed = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return seed, public


def test_sign_refuses_when_no_key_is_supplied(monkeypatch):
    """There must be no default key.

    A signature scheme whose key is in the repository authenticates nothing,
    because anyone holding the repository can sign.
    """
    monkeypatch.delenv("MECH_SIGNING_KEY_PATH", raising=False)
    with pytest.raises(SigningUnavailable):
        sign("payload")


def test_signature_verifies_with_the_public_key_alone():
    seed, public = _keypair()
    signature = sign("payload", seed)

    result = verify("payload", signature, public)
    assert result.valid
    assert result.reason is None
    assert result.algorithm == "Ed25519"


def test_forged_signature_and_tampered_payload_both_fail():
    """The property the old scheme lacked: verification must need no secret."""
    seed, public = _keypair()
    signature = sign("payload", seed)

    assert not verify("payload", "00" * 32, public).valid
    assert not verify("payload", signature, _keypair()[1]).valid
    assert not verify("payload" + "x", signature, public).valid


def test_missing_signature_and_missing_key_are_distinguishable_from_invalid():
    """A bare bool cannot tell these three apart, and callers eventually
    treat a missing signature as a valid one."""
    seed, public = _keypair()
    signature = sign("payload", seed)

    assert verify("payload", None, public).reason == "no signature is recorded"
    assert verify("payload", "", public).reason == "no signature is recorded"
    assert verify("payload", signature, None).reason == "no public key was supplied"

    wrong = verify("payload", "00" * 32, public)
    assert not wrong.valid
    assert "does not verify" in wrong.reason


def test_public_key_cannot_be_used_to_sign():
    """The decisive property. The old `verify_signature` recomputed with the
    literal secret, so it verified by *knowing* it."""
    _, public = _keypair()
    with pytest.raises(SigningUnavailable):
        sign("forged", public)


def test_raw_binary_keys_are_not_stripped():
    """Regression: `raw.strip()` ran before the length check.

    A raw 32-byte Ed25519 seed is binary. `strip()` removes bytes 0x09-0x0d and
    0x20 from either end, so a seed whose first or last byte was 0x20 arrived as
    31 bytes and was rejected as malformed. Roughly 2*6/256 = 4.7% of randomly
    generated keys, which surfaced as an intermittent ~1-in-8 test failure.

    Deterministic here: seeds are built with the offending byte at each end.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    for edge_byte in (0x20, 0x09, 0x0A, 0x0D):
        seed = bytearray(Ed25519PrivateKey.generate().private_bytes_raw())
        seed[0] = edge_byte
        seed[31] = edge_byte
        raw = bytes(seed)

        assert len(raw) == 32
        signature = sign("payload", raw)
        assert len(signature) == 128

        public = Ed25519PrivateKey.from_private_bytes(raw).public_key().public_bytes_raw()
        assert verify("payload", signature, public).valid

        # The 64-byte expanded form (seed || public-seed) works directly.
        expanded = bytes(seed) + bytes(seed[:32])
        assert len(expanded) == 64
        assert verify("payload", sign("payload", expanded), public).valid

        # Binary material padded with whitespace is ambiguous -- `strip()` also
        # eats whitespace bytes just inside the ends -- so it must be rejected
        # rather than silently interpreted as a different key. PEM and hex are
        # the padded/keyed forms that are actually supported.
        padded = bytes([edge_byte]) + expanded + bytes([edge_byte])
        with pytest.raises(SigningUnavailable):
            sign("payload", padded)

        # The supported textual forms, with surrounding whitespace, do work.
        assert verify("payload", sign("payload", seed.hex()), public).valid
        assert verify("payload", sign("payload", f"  {seed.hex()}  "), public).valid


def test_hex_seed_is_not_mistaken_for_64_raw_bytes():
    """Regression from the same fix, opposite direction.

    A 64-character hex seed is also 64 *bytes*. With a length-first order it was
    handed to `from_private_bytes` as binary, so signing succeeded under a key the
    caller never supplied -- a valid signature attesting to the wrong identity,
    which is worse than a hard failure.
    """
    seed, public = _keypair()
    hex_seed = seed.hex()
    assert len(hex_seed) == 64, "a 32-byte seed hex-encodes to 64 characters"

    # Must be the *same* signature as the raw bytes, i.e. the same key.
    assert sign("payload", hex_seed) == sign("payload", seed)
    assert verify("payload", sign("payload", hex_seed), public).valid

    # And the raw public key, plus its hex form, must both verify. (Not
    # `public.hex()` -- `public` is PEM, so hex-encoding it would encode the ASCII
    # armour rather than the key.)
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    raw_public = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes_raw()
    signature = sign("payload", seed)
    assert verify("payload", signature, raw_public).valid
    assert verify("payload", signature, raw_public.hex()).valid
    assert verify("payload", signature, f"  {raw_public.hex()}  ").valid

    # A 64-character hex *expanded* private key (seed || public-seed) resolves to
    # the same key as the raw 64-byte form.
    expanded = seed + raw_public
    assert len(expanded) == 64
    assert sign("payload", expanded.hex()) == sign("payload", expanded)
    assert sign("payload", expanded.hex()) == signature


def test_signing_key_can_come_from_env(monkeypatch):
    seed, public = _keypair()
    monkeypatch.setenv("MECH_SIGNING_KEY_PATH", "")
    monkeypatch.delenv("MECH_SIGNING_KEY_PATH")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as handle:
        handle.write(seed)
        path = handle.name
    try:
        from_arg = sign("payload", path)
        monkeypatch.setenv("MECH_SIGNING_KEY_PATH", path)
        assert sign("payload") == from_arg
        assert verify("payload", from_arg, public).valid
    finally:
        os.unlink(path)


# ── No signature-shaped hash anywhere ──────────────────────────────────────

def test_no_keyed_hash_presented_as_a_signature():
    """Guards the exact anti-pattern: sha256 of a payload plus a key literal.

    Matched on call syntax rather than a loose pattern, because the module
    docstrings in these files *describe* the old construction, and a regex
    loose enough to catch the code also catches the explanation of why the
    code was wrong.
    """
    targets = [
        "backend/research_datasets/dataset_manager.py",
        "backend/science/reproducibility/research_manifest.py",
        "backend/science/integrity/signing.py",
    ]
    offenders = []
    for rel in targets:
        path = Path(__file__).resolve().parents[2] / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "sha256"):
                continue
            arg = node.args[0] if node.args else None
            if isinstance(arg, ast.JoinedStr):
                joined = "".join(
                    v.value for v in arg.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                )
                joined += "".join(
                    v.value.id if isinstance(v, ast.Name) else ""
                    for v in arg.values
                    if isinstance(v, ast.FormattedValue)
                    and isinstance(v.value, ast.Name)
                )
                if joined.endswith(":") and "key" in joined.lower():
                    offenders.append(f"{rel}:{node.lineno} sha256(payload:key)")

    assert not offenders, "keyed hash standing in for a signature: " + "; ".join(offenders)


def test_verify_signature_signature_takes_a_required_public_key():
    """`verify_signature(public_key="mock_public_key")` ignored its own
    argument. The key must now be positional and required, so a caller cannot
    reach verification without thinking about which key."""
    import inspect

    from backend.research_datasets.dataset_manager import DatasetManager as DM

    params = inspect.signature(DM.verify_signature).parameters
    assert "public_key" in params
    assert params["public_key"].default is inspect.Parameter.empty

    params = inspect.signature(DM.sign_dataset).parameters
    assert params["private_key"].default is None  # None => env var, never a literal


def test_no_literal_signing_key_remains_in_source():
    """A secret committed to the tree is the whole defect in one grep."""
    import re

    pattern = re.compile(r"(mock_private_key|mock_public_key|mock_key)")
    for rel in ("backend/research_datasets/dataset_manager.py",
                "backend/science/reproducibility/research_manifest.py"):
        path = Path(__file__).resolve().parents[2] / rel
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.keyword,)):
                for value in (node.value,):
                    for sub in ast.walk(value):
                        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                            assert not pattern.search(sub.value), (
                                f"{rel}:{node.lineno} default signing key present")


# ── Dataset integrity ──────────────────────────────────────────────────────

def test_placeholder_digests_are_recognised_as_unrecorded():
    """`golden_manifest.json` records bundle_hash as sha256(32 zero bytes) and
    prompt_hash as sha256(""). Both mean "nothing recorded", not "mismatch"."""
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    zeros = "d41d8cd98f00b204e9800998ecf8427e"
    assert _is_placeholder(empty)
    assert _is_placeholder(zeros)
    assert _is_placeholder("")
    assert _is_placeholder("sha256:token_hash_placeholder".replace("sha256:", ""))
    assert not _is_placeholder("a" * 64)


def test_load_reports_which_hashes_actually_ran():
    """The old docstring said "Triple-SHA" while comparing one hash."""
    manager = DatasetManager(data_dir="backend/research_datasets")
    manager.load("ioi")

    status = manager.last_integrity_status
    assert set(status["checks"]) >= {"bundle_hash", "prompt_hash"}
    # Placeholders are reported as not_recorded, never as verified.
    for name, state in status["checks"].items():
        assert state in {"verified", "not_recorded", "mismatch"}, name
        if state == "verified":
            assert not _is_placeholder(str(status["checks"][name]))


def test_tampered_dataset_raises_and_bypass_is_named():
    import shutil
    import tempfile

    src = Path(__file__).resolve().parents[2] / "backend" / "research_datasets"
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("ioi", "golden_manifest.json"):
            source = src / name
            target = Path(tmp) / name
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)

        dataset_file = Path(tmp) / "ioi" / "dataset.json"
        data = json.loads(dataset_file.read_text(encoding="utf-8"))
        data["prompts"].append({"id": "injected", "clean": "x", "corrupted": "y",
                                "target": " z", "metadata": {}})
        dataset_file.write_text(json.dumps(data), encoding="utf-8")

        # Record a real hash for the *original* prompts, so the only way to
        # detect the injected prompt is to actually compare.
        import hashlib

        manifest_file = Path(tmp) / "golden_manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        original = json.loads((src / "ioi" / "dataset.json").read_text(encoding="utf-8"))
        real_prompt_hash = hashlib.sha256(
            json.dumps(original["prompts"], sort_keys=True).encode("utf-8")
        ).hexdigest()
        real_bundle = hashlib.sha256(
            (src / "ioi" / "dataset.json").read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        for key, value in (("prompt_hash", real_prompt_hash),
                           ("bundle_hash", real_bundle)):
            manifest["datasets"]["IOI-Canonical-100"]["hashes"][key] = f"sha256:{value}"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        manager = DatasetManager(data_dir=tmp)
        with pytest.raises(ValueError, match="integrity check failed"):
            manager.load("ioi")
        assert manager.last_integrity_status["integrity_verified"] is False

        # The escape hatch must be unmistakably named and leave a trace.
        os.environ["MECH_INTEGRITY_CHECKS"] = "disabled-for-local-development"
        try:
            manager.load("ioi")
            assert manager.last_integrity_status["bypassed"] is True
            assert manager.last_integrity_status["integrity_verified"] is False
        finally:
            del os.environ["MECH_INTEGRITY_CHECKS"]

        


def test_old_hash_bypass_env_var_is_gone():
    source = (Path(__file__).resolve().parents[2]
              / "backend/research_datasets/dataset_manager.py").read_text(encoding="utf-8")
    executable = executable_source(source)
    assert "MECH_BYPASS_HASH_CHECK" not in executable
    assert "MECH_BYPASS_HASH_CHECK" not in executable_source(
        (Path(__file__).resolve().parents[2]
         / "backend/science/reproducibility/run_reproducibility_demo.py")
        .read_text(encoding="utf-8")
    )


# ── Statistics and status ──────────────────────────────────────────────────

def test_unmeasured_patch_success_cannot_pass():
    """`patch_success_rate=100.0` meant the criterion that could not be
    checked was the one that passed."""
    validator = ScientificValidator()
    observations = [1.0] * 40 + [0.0] * 5   # mean 0.889, comfortably in range

    with pytest.raises(ValueError, match="no observations were supplied"):
        validator.validate_benchmark("ioi_faithfulness", [])

    stats = validator.validate_benchmark(
        "ioi_faithfulness", observations, patch_success_rate=None
    )
    assert stats.criteria_met["patch_success_met"] == "not_assessed"
    assert stats.verdict != "PASS"


def test_manifest_status_is_derived_not_asserted():
    assert derive_manifest_status("PASS", 95.0, signed=True) == "VALIDATED"
    assert derive_manifest_status("PASS", 95.0, signed=False) == "UNSIGNED"
    assert derive_manifest_status("PASS", 55.0, signed=True) == "REVISION_REQUIRED"
    assert derive_manifest_status("FAIL", 95.0, signed=True) == "UNVALIDATED"
    assert derive_manifest_status("", 95.0, signed=True) == "UNVALIDATED"
    assert derive_manifest_status("REVISION_REQUIRED", 95.0, signed=True) == "REVISION_REQUIRED"


def test_signing_never_upgrades_a_failed_manifest():
    from dataclasses import dataclass

    @dataclass
    class _Snap:
        env: str = "test"

    seed, public = _keypair()
    engine = ResearchManifestEngine()
    manifest = engine.generate("EXP", _Snap(), {"b": 2}, [{"c": 3}],
                               reproducibility_score=99.0, verdict="FAIL")
    assert manifest.status == "UNVALIDATED"

    engine.sign_manifest(manifest, seed)
    assert manifest.status == "UNVALIDATED", "signing upgraded a failed benchmark"
    assert engine.verify_manifest_signature(manifest, public)["valid"]


def test_certificate_does_not_default_to_a_literature_value():
    """`"published": 0.88,  # Mocked lookup` became `results.get("published")`."""
    from dataclasses import asdict

    from backend.science.reproducibility.benchmark_certificate import CertificateEngine

    cert = CertificateEngine().generate(
        benchmark_id="ioi",
        results={"current": 0.724},
        hashes={},
        repro_score=55.0,
        verdict=False,
    )
    data = asdict(cert)
    assert data["published_reference"] is None
    assert data["regression"] is None


# ── The demo script ────────────────────────────────────────────────────────

#: Directories on the measurement and validation path. A series built by formula
#: in any of these would be a fabricated measurement.
_VALIDATION_DIRS = (
    "backend/science/reproducibility",
    "backend/interpretability/discovery",
    "backend/validation",
    "backend/agents",
)


def test_validation_path_does_not_synthesise_observations():
    """No module on the measurement or validation path may build a series of
    "observations" from a formula.

    The old audit generated `[0.88 + (0.02 * (0.5 - i/100.0)) for i in range(100)]`
    and fed it to the statistical validator. The bootstrap CI, Cohen's d and the
    verdict were all computed over an arithmetic ramp, so the PASS was guaranteed
    by construction and reported as a finding.

    Scoped to the validation path on purpose. Plenty of modules elsewhere build
    synthetic series -- mock attention banks, seeded stand-ins -- and those are
    legitimate *as fixtures* provided they are labelled. `dispatcher.py`'s seeded
    attention maps carry `provenance: "seeded"` and a note saying they are "not
    model measurements"; that is the correct pattern. The defect is a formula
    series reaching a validator, not the existence of formula series.
    """
    root = Path(__file__).resolve().parents[2]
    offenders: List[str] = []
    for rel_dir in _VALIDATION_DIRS:
        directory = root / rel_dir
        if not directory.exists():
            continue
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ListComp):
                    continue
                src = ast.unparse(node)
                if "range(" not in src or "+" not in src:
                    continue
                floats = [n.value for n in ast.walk(node)
                          if isinstance(n, ast.Constant) and isinstance(n.value, float)]
                if len(floats) >= 2:
                    offenders.append(
                        f"{path.relative_to(root)}:{node.lineno} {src[:70]}")
    assert not offenders, "synthesised series on the validation path: " + "; ".join(offenders)


def test_nothing_formula_generated_is_labelled_a_measurement():
    """The inverse hazard: a series built by formula must not carry a
    provenance label claiming it was observed."""
    root = Path(__file__).resolve().parents[2]
    offenders: List[str] = []
    for path in (root / "backend").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            if "provenance" not in keys:
                continue
            provenance = next(
                (v.value for k, v in zip(node.keys, node.values)
                 if isinstance(k, ast.Constant) and k.value == "provenance"
                 and isinstance(v, ast.Constant)),
                None,
            )
            if provenance in {"live", "model_logits", "observed"}:
                # A dict claiming live provenance must not also be built
                # entirely from float literals.
                floats = [n.value for n in ast.walk(node)
                          if isinstance(n, ast.Constant) and isinstance(n.value, float)]
                if len(floats) >= 2:
                    offenders.append(
                        f"{path.relative_to(root)}:{node.lineno} provenance={provenance}")
    assert not offenders, "formula-built data labelled live: " + "; ".join(offenders)


# ── Dataset identity and promotion ─────────────────────────────────────────

def test_both_dataset_ids_resolve_to_the_same_file():
    """`load` built its path from the id it was given, so the manifest's own
    canonical id raised FileNotFoundError and only the folder alias worked."""
    manager = DatasetManager(data_dir="backend/research_datasets")
    assert len(manager.load("IOI-Canonical-100")) == len(manager.load("ioi"))


def test_signature_is_independent_of_which_id_was_used():
    """The signing payload embedded the caller's spelling.

    `golden_manifest.json` registers the dataset as `IOI-Canonical-100` while
    the file lives in `datasets/ioi/`. With both spellings resolving to the same
    dataset, a payload built from the caller's id meant signing via the alias and
    verifying via the canonical id disagreed -- a signature that could not be
    checked without knowing which spelling the signer happened to use.
    """
    manager = DatasetManager(data_dir="backend/research_datasets")
    meta = manager._manifest["ioi"]

    by_alias = manager._signing_payload("ioi", meta)
    by_canonical = manager._signing_payload("IOI-Canonical-100", meta)
    assert by_alias == by_canonical
    assert by_alias.startswith("IOI-Canonical-100:")

    # And end to end: sign under one spelling, verify under the other.
    seed, public = _keypair()
    manager.sign_dataset("ioi", seed)
    assert manager.verify_signature("IOI-Canonical-100", public)["valid"]


def _dataset_store(tmp_path, record_hashes: bool):
    """A copy of the dataset store, optionally with real recorded hashes."""
    import hashlib
    import shutil

    src = Path(__file__).resolve().parents[2] / "backend" / "research_datasets"
    dst = tmp_path / "research_datasets"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))

    if record_hashes:
        raw = (dst / "ioi" / "dataset.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        manifest_path = dst / "golden_manifest.json"
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = document["datasets"]["IOI-Canonical-100"]["hashes"]
        hashes["bundle_hash"] = "sha256:" + hashlib.sha256(
            raw.encode("utf-8")).hexdigest()
        hashes["prompt_hash"] = "sha256:" + hashlib.sha256(
            json.dumps(data["prompts"], sort_keys=True).encode("utf-8")).hexdigest()
        manifest_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    return dst


def _key_files(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    private = tmp_path / "signing.pem"
    private.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    public = tmp_path / "verify.pem"
    public.write_bytes(key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    other = tmp_path / "other.pem"
    other.write_bytes(Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    return str(private), str(public), str(other)


def _load_script(name: str, relative: str):
    import importlib.util

    path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_promotion_refuses_without_a_key(tmp_path, capsys):
    """It used to call `sign_dataset(dataset_id)` with no key, sign with the
    literal secret from the source, and print "Signature Applied"."""
    store = _dataset_store(tmp_path, record_hashes=True)
    private, _, _ = _key_files(tmp_path)
    script = _load_script("promote_ds", "frontend/scripts/promote_dataset.py")
    script.DATA_DIR = str(store)

    assert script.promote_dataset("ioi", "tester") == 1
    assert script.promote_dataset("ioi", "tester", "") == 1
    out = capsys.readouterr().out
    assert "no default signing key" in out
    assert "Signature Applied" not in out


def test_promotion_refuses_when_manifest_records_no_hash(tmp_path, capsys):
    """GOLDEN means attested. With placeholder hashes there is nothing to attest."""
    store = _dataset_store(tmp_path, record_hashes=False)
    private, _, _ = _key_files(tmp_path)
    script = _load_script("promote_ds", "frontend/scripts/promote_dataset.py")
    script.DATA_DIR = str(store)

    assert script.promote_dataset("ioi", "tester", private) == 1
    assert "GOLDEN requires recorded" in capsys.readouterr().out

    manifest = json.loads((store / "golden_manifest.json").read_text(encoding="utf-8"))
    assert manifest["datasets"]["IOI-Canonical-100"]["status"] == "GOLDEN"


def test_promotion_persists_the_signature_to_disk(tmp_path, capsys):
    """Regression: the signature and GOLDEN status were assigned to an in-memory
    dict and never written back, so the script reported "PROMOTION COMPLETE ...
    GOLDEN and signed" while the manifest on disk kept its placeholder
    signature. Every later verification ran against a manifest promotion had
    never touched."""
    store = _dataset_store(tmp_path, record_hashes=True)
    private, public, _ = _key_files(tmp_path)

    promote = _load_script("promote_ds", "frontend/scripts/promote_dataset.py")
    promote.DATA_DIR = str(store)
    assert promote.promote_dataset("ioi", "tester", private) == 0
    capsys.readouterr()

    on_disk = json.loads((store / "golden_manifest.json").read_text(encoding="utf-8"))
    entry = on_disk["datasets"]["IOI-Canonical-100"]
    assert entry["signature_algorithm"] == "Ed25519"
    assert len(entry["signature"]) == 128, "a hex Ed25519 signature is 64 bytes"
    assert entry["promoted_by"] == "tester"

    # The root_signature placeholder must not survive next to a real signature.
    assert "placeholder" not in on_disk["root_signature"]
    import hashlib
    assert on_disk["root_signature"] == "sha256:" + hashlib.sha256(
        json.dumps(on_disk["datasets"], sort_keys=True).encode("utf-8")).hexdigest()

    # And it must verify from a fresh manager reading the file from disk.
    verify_script = _load_script("verify_ds", "frontend/scripts/verify_golden_datasets.py")
    verify_script.DATA_DIR = str(store)
    monkey = os.environ.get("MECH_VERIFY_PUBLIC_KEY_PATH")
    os.environ["MECH_VERIFY_PUBLIC_KEY_PATH"] = public
    try:
        assert verify_script.audit_datasets() == 0
    finally:
        if monkey is None:
            del os.environ["MECH_VERIFY_PUBLIC_KEY_PATH"]
        else:
            os.environ["MECH_VERIFY_PUBLIC_KEY_PATH"] = monkey


def test_verifier_needs_a_public_key_and_separates_unrecorded(tmp_path, capsys, monkeypatch):
    store = _dataset_store(tmp_path, record_hashes=False)
    script = _load_script("verify_ds", "frontend/scripts/verify_golden_datasets.py")
    script.DATA_DIR = str(store)

    monkeypatch.delenv("MECH_VERIFY_PUBLIC_KEY_PATH", raising=False)
    assert script.audit_datasets() == 1
    out = capsys.readouterr().out
    assert "not_recorded" in out
    assert "manifest is incomplete" in out

    private, public, other = _key_files(tmp_path)

    # Real hashes, real matching key, but nothing signed: must not claim success.
    store2 = _dataset_store(tmp_path / "b", record_hashes=True)
    script.DATA_DIR = str(store2)
    monkeypatch.setenv("MECH_VERIFY_PUBLIC_KEY_PATH", public)
    assert script.audit_datasets() == 1
    assert "does not verify" in capsys.readouterr().out

    # Sign it, then the same public key must succeed.
    promote = _load_script("promote_ds", "frontend/scripts/promote_dataset.py")
    promote.DATA_DIR = str(store2)
    assert promote.promote_dataset("ioi", "tester", private) == 0
    capsys.readouterr()
    assert script.audit_datasets() == 0
    capsys.readouterr()

    # A different public key must fail.
    monkeypatch.setenv("MECH_VERIFY_PUBLIC_KEY_PATH", other)
    assert script.audit_datasets() == 1


def test_no_script_catches_an_integrity_failure_and_reloads_bypassed(tmp_path):
    """The old audit's shape: `except` -> set a bypass -> load again.

    Disabling a check and retrying turns a detected integrity failure into a
    successful load, and because the bypass was an environment variable it
    affected every later load in the process.
    """
    root = Path(__file__).resolve().parents[2]
    offenders: List[str] = []
    for script in ("frontend/scripts/promote_dataset.py",
                   "frontend/scripts/verify_golden_datasets.py",
                   "backend/science/reproducibility/run_reproducibility_demo.py"):
        path = root / script
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            body = ast.unparse(node.body)
            if "environ" in body or "MECH_INTEGRITY_CHECKS" in body:
                offenders.append(f"{script}:{node.lineno}")
    assert not offenders, "integrity bypass inside an exception handler: " + "; ".join(offenders)


def test_certificate_does_not_present_a_placeholder_as_a_signature():
    """`signature` was copied straight from the manifest, which ships the literal
    `sha256:dataset_sig_placeholder`. The certificate therefore issued a
    placeholder as a signature."""
    import copy
    from dataclasses import asdict

    from backend.research_datasets.dataset_certificate import DatasetCertificateEngine

    manager = DatasetManager(data_dir="backend/research_datasets")
    shipped = copy.deepcopy(manager._manifest["ioi"])
    assert shipped["signature"] == "sha256:dataset_sig_placeholder"

    cert = asdict(DatasetCertificateEngine().issue(shipped))
    assert cert["signature"] == ""
    assert cert["signature_status"] == "unsigned"
    # The empty-digest hashes are placeholders behind a `sha256:` prefix and must
    # not be reported as recorded either.
    assert cert["prompt_hash"] is None
    assert cert["bundle_hash"] is None
    assert cert["token_hash"] is None
    assert cert["integrity_status"] == "no_hashes_recorded"
    # A certificate must not carry a verification time for a check that never ran.
    assert cert["verified_at"] is None


def test_certificate_distinguishes_recorded_unsigned_and_legacy_signature():
    import copy
    from dataclasses import asdict

    from backend.research_datasets.dataset_certificate import DatasetCertificateEngine

    manager = DatasetManager(data_dir="backend/research_datasets")
    base = copy.deepcopy(manager._manifest["ioi"])
    base["hashes"] = {"prompt_hash": "sha256:" + "a" * 64,
                      "token_hash": "sha256:" + "b" * 64,
                      "bundle_hash": "sha256:" + "c" * 64}
    engine = DatasetCertificateEngine()

    unsigned = asdict(engine.issue(base))
    assert unsigned["integrity_status"] == "recorded_not_checked"
    assert unsigned["signature_status"] == "unsigned"
    assert unsigned["unrecorded_fields"] == []

    signed = copy.deepcopy(base)
    signed["signature"] = "d" * 128
    signed["signature_algorithm"] = "Ed25519"
    result = asdict(engine.issue(signed))
    assert result["signature_status"] == "present_Ed25519_not_checked_here"

    # A 64-hex-char "signature" is the old keyed SHA-256, which cannot be verified.
    legacy = copy.deepcopy(base)
    legacy["signature"] = "a" * 64
    assert asdict(engine.issue(legacy))["signature_status"] == "unverifiable_algorithm"


def test_placeholder_hash_behind_a_sha256_prefix_is_still_a_placeholder():
    """Every hash in `golden_manifest.json` carries an `sha256:` prefix.

    Without stripping it, `sha256:e3b0c442...` (the empty-string digest) was not
    recognised, so the dataset scored as having recorded that hash.
    """
    from backend.research_datasets.dataset_manager import _is_placeholder

    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert _is_placeholder(f"sha256:{empty}")
    assert _is_placeholder("sha256:dataset_sig_placeholder")
    assert _is_placeholder(empty)
    assert not _is_placeholder("sha256:" + "a" * 64)


def test_health_score_does_not_credit_placeholders():
    """Two optimistic defaults, both of which scored the shipped dataset well.

    * `integrity` was `1.0 if all(k in h for k in [...])` -- it checked only that
      the three keys existed. All three do, but two are placeholder digests, so
      the dataset scored 100% on integrity while recording nothing verifiable.
    * `sig_score` was `1.0 if meta.get("signature")` -- and the manifest's
      signature is the truthy string `sha256:dataset_sig_placeholder`, so the
      dataset scored 100% for being signed.
    """
    manager = DatasetManager(data_dir="backend/research_datasets")
    health = manager.compute_health_score("ioi")

    assert health["integrity_hashes_recorded"] == "0/3"
    assert sorted(health["integrity_unrecorded"]) == [
        "bundle_hash", "prompt_hash", "token_hash"]
    assert health["integrity"] == 0.0

    assert health["signature_is_real"] is False
    assert health["signature"] == 0.0


def test_badge_never_asserts_signed_unconditionally():
    """`_generate_badge` returned the literal `Status: **GOLDEN & SIGNED**` for
    every dataset, and `export` defaulted a missing health method to
    `{"overall": 100}` -- so a perfect badge was reachable with nothing checked."""
    from backend.research_datasets.dataset_exporter import DatasetExporter

    exporter = DatasetExporter()

    unsigned = exporter._generate_badge(100.0, {
        "integrity_hashes_recorded": "0/3", "signature_is_real": False,
        "overall_is_partial": True, "overall_weight_covered": 0.9,
        "overall_unmeasured_weight": 0.1,
    })
    assert "GOLDEN & SIGNED" not in unsigned
    assert "UNVERIFIED" in unsigned

    partial = exporter._generate_badge(80.0, {
        "integrity_hashes_recorded": "2/3", "signature_is_real": False})
    assert "UNSIGNED" in partial
    assert "80.0%" in partial

    signed = exporter._generate_badge(95.0, {
        "integrity_hashes_recorded": "3/3", "signature_is_real": True})
    assert "SIGNED" in signed and "UNSIGNED" not in signed

    # No health computed at all: reported as unknown, not as 100%.
    unknown = exporter._generate_badge(None, {})
    assert "not computed" in unknown
    assert "100%" not in unknown


def test_exporter_docstring_lists_only_files_it_writes():
    """The docstring advertised certificate.json, CHANGELOG.md and citation.bib.
    None was ever written, so it described a bundle that did not exist."""
    source = doc_and_comments(
        (Path(__file__).resolve().parents[2]
         / "backend/research_datasets/dataset_exporter.py").read_text(encoding="utf-8")
    )
    tree = ast.parse(source)
    exported = set()
    for node in ast.walk(tree):
        # Dict literal: {"dataset.json": ...}
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    exported.add(key.value)
        # Subscript assignment: files_to_bundle["bundle_manifest.json"] = ...
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].slice, ast.Constant)
                and isinstance(node.targets[0].slice.value, str)):
            exported.add(node.targets[0].slice.value)
    for name in ("CITATION.cff", "dataset.json", "metadata.json",
                 "REPRODUCIBILITY_BADGE.md", "verify_bundle.py"):
        assert name in exported, f"{name} is not actually written"

    # The docstring's *listing* (its bullet lines) must match what is written.
    # Checked on the bullets only: the surrounding prose names the three files
    # that were never produced, in order to say so.
    module_docstring = ast.get_docstring(tree) or ""
    listing = [line.strip("- ").strip() for line in module_docstring.splitlines()
               if line.strip().startswith("- ")]
    assert listing, "expected a bulleted file listing in the module docstring"
    for claimed in listing:
        # Bullets may carry a parenthetical gloss ("dataset.json (prompts)").
        name = claimed.split(" (")[0].strip()
        assert name in exported, (
            f"docstring lists {claimed!r}, which export() never writes")
    assert not any(c in listing for c in ("CHANGELOG.md", "citation.bib",
                                          "certificate.json"))


def test_certificate_hash_fields_do_not_hold_identifiers():
    """`dataset_hash` held a dataset id and `environment_hash` a snapshot id.

    Both are identifiers stored under names ending in `_hash`, and absent fields
    became the string `"unknown"` -- indistinguishable from a recorded digest
    once the value is in a certificate.
    """
    from dataclasses import asdict

    from backend.science.reproducibility.benchmark_certificate import CertificateEngine

    cert = asdict(CertificateEngine().generate(
        benchmark_id="ioi",
        results={"current": 0.724},
        hashes={"dataset_id": "ioi", "environment_id": "snap_1",
                "model": "sha256:" + "a" * 64, "tokenizer": "unknown"},
        repro_score=55.0,
        verdict=False,
    ))

    assert cert["dataset_id"] == "ioi"
    assert cert["environment_id"] == "snap_1"
    # Identifiers must not masquerade as digests.
    assert cert["dataset_hash"] is None
    assert cert["environment_hash"] is None
    assert cert["tokenizer_hash"] is None
    assert cert["model_sha256"] == "a" * 64

    for field in ("dataset_hash", "tokenizer_hash", "model_sha256",
                  "environment_hash", "published_reference", "regression"):
        assert cert[field] is not "unknown"


def test_old_audit_script_is_gone_and_demo_declares_itself():
    root = Path(__file__).resolve().parents[2]
    audit = root / "backend/science/reproducibility/run_reproducibility_audit.py"
    assert not audit.exists(), "the fabricating audit script still exists"

    demo = root / "backend/science/reproducibility/run_reproducibility_demo.py"
    source = demo.read_text(encoding="utf-8")
    assert "NOT A SCIENTIFIC VALIDATION" in source

    # It must not force mock mode or disable an integrity check anywhere.
    executable = executable_source(source)
    assert "mock_mode=True" not in executable
    assert "MECH_INTEGRITY_CHECKS" not in executable


def test_demo_runs_real_benchmarks_without_inventing_results():
    """The demo's contract: real weights or raise. Never a fabricated fallback."""
    root = Path(__file__).resolve().parents[2]
    source = (root / "backend/science/reproducibility/run_reproducibility_demo.py").read_text(
        encoding="utf-8")
    assert "BenchmarkRunner(mock_mode=False)" in source
    # No bare `except` that swallows a failure and carries on with data.
    assert "except Exception" not in executable_source(source)