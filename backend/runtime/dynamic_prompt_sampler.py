"""Dynamic Probe Sampler for MECH.

Generates a fresh, randomly-sampled probe set every time the app opens.
No two sessions share the same fixed prompts.

Draws from template pools across five behavioral categories:
    factual_recall   — geography, science, history
    arithmetic       — basic numeric completions
    relational       — subject-predicate-object chains
    ioi              — indirect object identification templates
    induction        — repeated-pattern continuations

Seeds from time.time() so every session is unique.
Writes the sampled set to session_probes.json in the OS temp dir.
The app reads this file at startup.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DynamicProbe:
    """A single dynamically-sampled probe."""
    probe_id: str
    category: str
    clean_prompt: str
    target_token: str
    corrupted_prompt: str
    distractor_token: str
    target_layer_fraction: float = 0.66
    target_neuron_idx: int = 412

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Template pools ────────────────────────────────────────────────────────────

_FACTUAL_TEMPLATES: List[Tuple[str, str, str, str]] = [
    # (clean_prompt, target_token, corrupted_prompt, distractor_token)
    ("The capital of France is", " Paris", "The capital of Italy is", " Rome"),
    ("The capital of Germany is", " Berlin", "The capital of Austria is", " Vienna"),
    ("The capital of Japan is", " Tokyo", "The capital of China is", " Beijing"),
    ("The capital of Spain is", " Madrid", "The capital of Portugal is", " Lisbon"),
    ("The capital of Australia is", " Canberra", "The capital of New Zealand is", " Wellington"),
    ("The capital of Canada is", " Ottawa", "The capital of Mexico is", " Mexico"),
    ("The capital of Brazil is", " Bras", "The capital of Argentina is", " Buenos"),
    ("The capital of Egypt is", " Cairo", "The capital of Morocco is", " Rabat"),
    ("The Eiffel Tower is located in", " Paris", "The Colosseum is located in", " Rome"),
    ("The Great Wall is located in", " China", "The Taj Mahal is located in", " India"),
    ("The official language of Spain is", " Spanish", "The official language of Germany is", " German"),
    ("The official language of France is", " French", "The official language of Italy is", " Italian"),
    ("The largest planet in our solar system is", " Jupiter", "The smallest planet in our solar system is", " Mercury"),
    ("Water freezes at", " 0", "Water boils at", " 100"),
    ("The first President of the United States was", " George", "The second President of the United States was", " John"),
]

_ARITHMETIC_TEMPLATES: List[Tuple[str, str, str, str]] = [
    ("Two plus two equals", " four", "Three plus three equals", " six"),
    ("Five plus five equals", " ten", "Four plus four equals", " eight"),
    ("Ten minus three equals", " seven", "Ten minus four equals", " six"),
    ("Three times three equals", " nine", "Two times four equals", " eight"),
    ("The square root of four is", " two", "The square root of nine is", " three"),
    ("One hundred divided by ten equals", " ten", "One hundred divided by five equals", " twenty"),
    ("Six plus seven equals", " thirteen", "Eight plus five equals", " thirteen"),
]

_RELATIONAL_TEMPLATES: List[Tuple[str, str, str, str]] = [
    ("Shakespeare wrote the play", " Hamlet", "Dickens wrote the novel", " Oliver"),
    ("Einstein developed the theory of", " general", "Newton discovered the law of", " gravity"),
    ("The Amazon river flows through", " Brazil", "The Nile river flows through", " Egypt"),
    ("The Pacific Ocean borders", " Asia", "The Atlantic Ocean borders", " Europe"),
    ("Mount Everest is located in", " Nepal", "Mount Fuji is located in", " Japan"),
    ("Beethoven composed the", " Ninth", "Mozart composed the", " Magic"),
]

_IOI_TEMPLATES: List[Tuple[str, str, str, str]] = [
    # Indirect object identification — the target is the IO, distractor is subject
    ("When Mary gave the book to John, John thanked", " Mary", "When Sarah gave the gift to Tom, Tom thanked", " Sarah"),
    ("After Alice sent the letter to Bob, Bob replied to", " Alice", "After Emma sent the email to James, James replied to", " Emma"),
    ("When the teacher handed the paper to the student, the student thanked", " the", "When the manager gave the report to the assistant, the assistant thanked", " the"),
    ("Lisa told Mark that she would call, and Mark waited for", " Lisa", "Julia told Chris that she would arrive, and Chris waited for", " Julia"),
]

_INDUCTION_TEMPLATES: List[Tuple[str, str, str, str]] = [
    # Repeated-token patterns — model should continue the pattern
    ("The sequence a b a b a b a", " b", "The sequence x y x y x y x", " y"),
    ("The pattern cat dog cat dog cat", " dog", "The pattern red blue red blue red", " blue"),
    ("1 2 3 1 2 3 1 2", " 3", "4 5 6 4 5 6 4 5", " 6"),
    ("alpha beta alpha beta alpha", " beta", "north south north south north", " south"),
]

_ALL_TEMPLATES = {
    "factual_recall": _FACTUAL_TEMPLATES,
    "arithmetic":     _ARITHMETIC_TEMPLATES,
    "relational":     _RELATIONAL_TEMPLATES,
    "ioi":            _IOI_TEMPLATES,
    "induction":      _INDUCTION_TEMPLATES,
}

# Canonical layer fractions and neuron indices per category
_CATEGORY_LAYER_FRACTIONS = {
    "factual_recall": 0.75,
    "arithmetic":     0.60,
    "relational":     0.70,
    "ioi":            0.66,
    "induction":      0.50,
}
_CATEGORY_NEURON_IDXS = {
    "factual_recall": 412,
    "arithmetic":     287,
    "relational":     501,
    "ioi":            384,
    "induction":      256,
}


# ── Sampler ───────────────────────────────────────────────────────────────────

def sample_session_probes(
    n_per_category: int = 2,
    seed: Optional[float] = None,
) -> List[DynamicProbe]:
    """
    Randomly samples n_per_category probes from each template pool.

    Parameters
    ----------
    n_per_category : probes drawn per behavioral category (default 2)
    seed           : explicit seed (default: current time in ms — unique per session)

    Returns
    -------
    List[DynamicProbe] — a fresh, randomized probe set for this session
    """
    rng = random.Random(seed if seed is not None else int(time.time() * 1000))

    probes: List[DynamicProbe] = []
    for category, templates in _ALL_TEMPLATES.items():
        chosen = rng.sample(templates, k=min(n_per_category, len(templates)))
        layer_frac  = _CATEGORY_LAYER_FRACTIONS[category]
        neuron_idx  = _CATEGORY_NEURON_IDXS[category]
        for i, (clean, target, corrupt, distractor) in enumerate(chosen):
            probe_id = f"probe_{category}_{i}_{rng.randint(1000, 9999)}"
            probes.append(DynamicProbe(
                probe_id=probe_id,
                category=category,
                clean_prompt=clean,
                target_token=target,
                corrupted_prompt=corrupt,
                distractor_token=distractor,
                target_layer_fraction=layer_frac,
                target_neuron_idx=neuron_idx,
            ))

    rng.shuffle(probes)
    return probes


def save_session_probes(probes: List[DynamicProbe], path: Optional[str] = None) -> str:
    """
    Serialises the session probe set to a JSON file.

    Returns
    -------
    Absolute path of the written file.
    """
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "mech_session_probes.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in probes], f, indent=2)
    return path


def load_session_probes(path: Optional[str] = None) -> List[DynamicProbe]:
    """Loads a previously saved session probe set from JSON."""
    if path is None:
        path = os.path.join(tempfile.gettempdir(), "mech_session_probes.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [DynamicProbe(**r) for r in raw]


def init_session(n_per_category: int = 2) -> Tuple[List[DynamicProbe], str]:
    """
    Top-level entry point called at app startup.

    Samples a fresh probe set, saves it, and returns (probes, file_path).
    Call this once at startup; subsequent code reads from the returned file.
    """
    probes = sample_session_probes(n_per_category=n_per_category)
    path   = save_session_probes(probes)
    return probes, path
