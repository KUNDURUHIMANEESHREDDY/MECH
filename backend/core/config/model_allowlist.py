"""Canonical allowlist for user-supplied model identifiers.

Endpoints that forward a user-supplied ``model_id``/``model_name`` into
model-loading calls (``from_pretrained`` / HuggingFace hub download) MUST
validate against this set to prevent SSRF / supply-chain abuse. A request
with an unrecognised model id is rejected instead of being passed verbatim
to the hub.

Keep this list in sync with the models actually supported by the platform
(see ``backend/api/dispatcher.py`` ``/models`` endpoint and the model
registry in ``backend/core/config/model_config.py``).
"""

from typing import Annotated

from pydantic import AfterValidator

ALLOWED_MODEL_IDS: frozenset[str] = frozenset(
    {
        "gpt2",
        "gpt2-small",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "distilgpt2",
        "gemma-2b",
        "gemma-7b",
        "pythia-1b",
        "llama-3-8b",
        "qwen-7b",
        "mistral-7b",
    }
)


def validate_model_id(value: str) -> str:
    if value not in ALLOWED_MODEL_IDS:
        raise ValueError(
            f"Model '{value}' is not an allowed model id. "
            f"Allowed ids: {sorted(ALLOWED_MODEL_IDS)}"
        )
    return value


ModelId = Annotated[str, AfterValidator(validate_model_id)]
