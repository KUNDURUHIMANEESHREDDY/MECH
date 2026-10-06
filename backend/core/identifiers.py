"""Identifier construction, with the two kinds of identity kept apart.

`hash()` is not a source of identifiers, and the code that used it that way had
three separate problems, which is why they are described one at a time rather
than as "use UUIDs".

**It is not stable across processes.** Python salts string hashing with
PYTHONHASHSEED, which is random per interpreter by default. So
`f"exp_{hash(str(payload)) % 10000}"` produced a different id for the same
payload on every run. A link, a citation or a replay written down yesterday
does not resolve today. In a system whose central claim is reproducibility,
anything derived from `hash()` is non-reproducible by construction, and no
amount of care elsewhere compensates for it.

**It is not unique.** `% 10000` is a ten-thousand-value namespace. By the
birthday bound two ids collide at around 100 items, so a collision is likely
well before the namespace is exhausted. These ids are persisted, so a collision
does not just lose a name -- it points two different experiments at one stored
row, and `delete_experiment` then removes whichever arrived first.

**It is not a hash.** `f"sha256_{hash(...) & 0xffffffff:08x}"` appeared
elsewhere in this codebase wearing a digest's name while being eight hex
characters of SipHash. Any consumer doing `len(x) == 64`, or treating the value
as collision-resistant, was being told something untrue.

The two kinds
-------------
These are different jobs and conflating them causes its own bugs, so there are
two functions rather than one with a flag:

* :func:`entity_id` -- "this is a new thing". Random, collision-free,
  appropriate for anything persisted or created per run. UUID4 rather than
  UUID7: UUID7's benefit is monotonic sortability, which nothing here consumes,
  and it needs a timestamp source this codebase does not otherwise have.
* :func:`content_id` -- "this is the same thing as that". Deterministic across
  processes and machines, appropriate for an id that *names* a piece of content
  so the same content gets the same name. Truncated SHA-256, not `hash()`,
  because it has to agree with itself tomorrow.

Choosing wrongly is a real cost in each direction: a `content_id` used for
entity identity collides whenever the content repeats, and an `entity_id` used
for content identity gives two identical results different names, so a
reproducibility check that compares ids can never pass.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

#: Hex characters kept from a SHA-256 digest. 32 hex characters is 128 bits,
#: which keeps the collision bound far beyond anything this system can store
#: while leaving ids readable. Truncation is a deliberate, documented choice --
#: unlike ``hash() & 0xffffffff``, which threw away 96 bits by accident.
CONTENT_ID_LENGTH = 32


def entity_id(prefix: str = "") -> str:
    """A fresh identifier for a new thing.

    Random rather than derived, because two runs of the same experiment are two
    different things and must be able to coexist. Use this for anything
    persisted.
    """
    token = uuid.uuid4().hex
    return f"{prefix}{token}" if prefix else token


def content_id(*parts: Any, prefix: str = "",
               length: int = CONTENT_ID_LENGTH) -> str:
    """A deterministic identifier derived from the content it names.

    The same parts always produce the same id, in this process and the next.
    Canonical JSON, so key order and separators cannot make two equal values
    hash differently.
    """
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"),
                           default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}" if prefix else digest