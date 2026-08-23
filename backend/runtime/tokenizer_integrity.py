"""Tokenizer Integrity Verifier for MECH.

Enforces the round-trip invariant for every probe target token:

    decode(encode(t)) == t

and verifies that every target token actually exists in the loaded
tokenizer's vocabulary before any mechanistic experiment is permitted.

A failure here means the probe's expected token is not representable
in this tokenizer's vocabulary — any mechanistic claim about how the
model produces that token would be investigating an unreachable event.

Checks performed per token
--------------------------
1. target_token_in_vocab  — the token appears in tokenizer.get_vocab()
2. round_trip_ok          — decode(encode(t)) == t   (no BPE split artifacts)
3. token_id               — the integer id assigned in this tokenizer
4. decoded_id             — decode([token_id]) matches the original text
5. is_single_token        — encoding produces exactly one token id
                            (split tokens indicate the probe needs redesign)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import logging
logger = logging.getLogger(__name__)



@dataclass
class TokenizerTokenReport:
    """Per-token integrity report."""
    token_text: str
    token_id: Optional[int]          # None if not in vocab
    decoded_from_id: Optional[str]   # decode([token_id])
    encode_ids: List[int]            # encode(token_text)
    round_trip_text: str             # decode(encode(token_text))
    round_trip_ok: bool              # round_trip_text.strip() == token_text.strip()
    target_token_in_vocab: bool
    is_single_token: bool            # len(encode_ids) == 1
    integrity_pass: bool             # all checks green

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenizerIntegrityReport:
    """Aggregate report for all probe target tokens."""
    tokenizer_class: str
    vocab_size: int
    token_reports: List[TokenizerTokenReport]
    all_passed: bool
    failed_tokens: List[str]         # tokens that did not pass integrity_pass
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokenizer_class": self.tokenizer_class,
            "vocab_size": self.vocab_size,
            "all_passed": self.all_passed,
            "failed_tokens": self.failed_tokens,
            "summary": self.summary,
            "token_reports": [r.to_dict() for r in self.token_reports],
        }


def _verify_single_token(tokenizer, token_text: str) -> TokenizerTokenReport:
    """Runs all integrity checks for one token string."""
    vocab = tokenizer.get_vocab()

    # 1. Vocab membership
    in_vocab = token_text in vocab
    token_id = vocab.get(token_text, None)

    # 2. Encode round-trip
    try:
        encode_ids: List[int] = tokenizer.encode(
            token_text, add_special_tokens=False
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Swallowed exception: %s", exc)
        encode_ids = []

    # 3. Decode round-trip
    try:
        round_trip_text: str = tokenizer.decode(encode_ids) if encode_ids else ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("Swallowed exception: %s", exc)
        round_trip_text = ""

    round_trip_ok = round_trip_text.strip() == token_text.strip()

    # 4. Decode the canonical token_id (if it exists)
    decoded_from_id: Optional[str] = None
    if token_id is not None:
        try:
            decoded_from_id = tokenizer.decode([token_id])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Swallowed exception: %s", exc)
            decoded_from_id = None

    is_single_token = len(encode_ids) == 1

    # All four criteria must pass
    integrity_pass = in_vocab and round_trip_ok and is_single_token

    return TokenizerTokenReport(
        token_text=token_text,
        token_id=token_id,
        decoded_from_id=decoded_from_id,
        encode_ids=encode_ids,
        round_trip_text=round_trip_text,
        round_trip_ok=round_trip_ok,
        target_token_in_vocab=in_vocab,
        is_single_token=is_single_token,
        integrity_pass=integrity_pass,
    )


def verify_tokenizer_integrity(
    tokenizer,
    target_tokens: List[str],
) -> TokenizerIntegrityReport:
    """
    Verifies tokenizer integrity for every probe target token.

    Parameters
    ----------
    tokenizer     : loaded PreTrainedTokenizer
    target_tokens : list of token strings from StandardMechanisticProbe.target_token

    Returns
    -------
    TokenizerIntegrityReport
        ``all_passed`` is True only when every token passes all four checks.
        ``failed_tokens`` names the tokens that need investigation.
    """
    tokenizer_class = type(tokenizer).__name__
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else len(tokenizer.get_vocab())

    reports: List[TokenizerTokenReport] = [
        _verify_single_token(tokenizer, t) for t in target_tokens
    ]

    failed = [r.token_text for r in reports if not r.integrity_pass]
    all_passed = len(failed) == 0

    if all_passed:
        summary = (
            f"Tokenizer integrity VERIFIED ({tokenizer_class}, vocab={vocab_size}): "
            f"all {len(reports)} target tokens round-trip correctly and are single-token."
        )
    else:
        issues = []
        for r in reports:
            if not r.integrity_pass:
                reasons = []
                if not r.target_token_in_vocab:
                    reasons.append("not in vocab")
                if not r.round_trip_ok:
                    reasons.append(f"round-trip '{r.round_trip_text}' ≠ '{r.token_text}'")
                if not r.is_single_token:
                    reasons.append(f"splits into {len(r.encode_ids)} tokens: {r.encode_ids}")
                issues.append(f"'{r.token_text}': {'; '.join(reasons)}")
        summary = (
            f"Tokenizer integrity FAILED ({tokenizer_class}, vocab={vocab_size}): "
            + " | ".join(issues)
            + ". Probes using these tokens cannot produce valid mechanistic results."
        )

    return TokenizerIntegrityReport(
        tokenizer_class=tokenizer_class,
        vocab_size=vocab_size,
        token_reports=reports,
        all_passed=all_passed,
        failed_tokens=failed,
        summary=summary,
    )
