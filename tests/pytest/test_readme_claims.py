"""The README must not claim more than the code does.

The rule under test
-------------------
**A document that describes a system is part of that system's provenance surface.**
A reader deciding whether to trust a number here has no way to check it against
the code, so a claim that has drifted from the implementation is worse than no
claim at all.

What drifted
------------
`README.md` described the Research Society as a

    ### Research Society - seven-agent workflow, all steps live

    `load -> reproduce -> inspect -> patch -> discover -> validate -> publish`,
    every step stamped `Provenance: live`, streamed to the UI over SSE.

Three claims, all false, in the most-read document in the repository:

1. **Seven agents.** `ResearchSocietyV2.__init__` constructs six:

       self.planner = Planner()
       self.executor = Executor()
       self.inspector = Inspector()
       self.discoverer = Discoverer()
       self.critic = Critic()
       self.scribe = Scribe()

   The likely origin of the error is that the *planner emits seven nodes* --
   load, reproduce, inspect, patch, discover, validate, publish. Seven steps were
   read as seven agents. The dispatcher at `society.py:96` names the same six.

2. **All steps live.** Measured on this machine for the IOI goal, only `reproduce`
   carries `provenance: live`:

       step        agent        status        provenance
       load        executor     loaded        unavailable
       reproduce   executor     completed     live
       inspect     inspector    ok            unavailable
       discover    discoverer   unavailable   live
       validate    critic       never ran     -
       publish     scribe       never ran     -

   The run reports `status: blocked`, `provenance: unavailable`,
   `validation_eligible: false`, `publication_eligible: false`, `4/5` completed.
   `validate` and `publish` never execute -- the workflow stops when `discover`
   fails.

3. **Every step stamped `Provenance: live`.** No run has ever produced that.

   The Critic's reflection is additionally a declared stub:
   `{"status": "unavailable", "provenance": "unavailable", "reason": "No live
   reflection executor is connected."}`, so that step cannot produce a
   measurement regardless of the run.

These tests guard the two claims that can be checked mechanically against the
code. The per-step table is not asserted line for line -- a README is prose, and
a test that fails on every rewording would get deleted rather than obeyed -- but
the reflection stub is required to be disclosed, because a declared stub presented
as a working step is the failure mode worth preventing.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
SOCIETY = ROOT / "backend" / "agents" / "society.py"

#: Phrases that assert more than the implementation supports. Each is a claim
#: that was measurably false at the time this file was written.
BANNED_CLAIMS = (
    "seven-agent",
    "seven agent",
    "7-agent",
    "all steps live",
    "every step stamped",
    "all seven agents",
)


def _code_agents() -> Set[str]:
    """Agent names `ResearchSocietyV2.__init__` actually constructs.

    Read from the AST rather than by importing, so the guard does not depend on
    torch being importable and cannot itself have side effects.
    """
    tree = ast.parse(SOCIETY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "ResearchSocietyV2":
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.FunctionDef) or sub.name != "__init__":
                continue
            names = set()
            for stmt in ast.walk(sub):
                # self.planner = Planner()
                if (isinstance(stmt, ast.Assign)
                        and isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)):
                    for target in stmt.targets:
                        if (isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == "self"):
                            names.add(target.attr)
            return names
    raise AssertionError("ResearchSocietyV2.__init__ not found in society.py")


def _society_section() -> str:
    """The README's Research Society section, up to the next heading."""
    text = README.read_text(encoding="utf-8")
    start = text.find("### Research Society")
    assert start != -1, "the README no longer has a Research Society section"
    rest = text[start + 3:]
    end = rest.find("\n### ")
    return rest[:end] if end != -1 else rest


def test_the_code_constructs_six_agents():
    """Precondition for the guard below: the count is what we think it is."""
    agents = _code_agents()
    assert agents == {"planner", "executor", "inspector",
                      "discoverer", "critic", "scribe"}, (
        f"ResearchSocietyV2 now constructs {sorted(agents)}. If that is "
        f"intentional, update this file and the README together -- the README "
        f"states the count and must not be left describing the old one.")


def test_the_dispatcher_agrees_with_the_constructor():
    """society.py names its agents in a second place.

    `dispatch()` resolves `explicit in ("planner", "executor", ...)` from a
    literal tuple. If that tuple and the constructor disagree, a node could be
    dispatched to an agent that does not exist.
    """
    source = SOCIETY.read_text(encoding="utf-8")
    match = re.search(
        r'explicit in \(([^)]*)\)', source, re.DOTALL)
    assert match, "could not find the agent-name tuple in society.py:dispatch"

    named = set(re.findall(r'"([a-z_]+)"', match.group(1)))
    assert named == _code_agents(), (
        f"dispatch() knows {sorted(named)} but __init__ builds "
        f"{sorted(_code_agents())}")


def test_the_readme_does_not_claim_more_agents_than_exist():
    section = _society_section()
    # The correction quotes the old wording in order to say it was wrong. Those
    # lines are the only places a banned phrase may appear.
    exemptions = ("was a count of steps", "Neither was true", "claimed")

    offenders = []
    for line in section.splitlines():
        low = line.lower()
        if any(exempt in line for exempt in exemptions):
            continue
        for phrase in BANNED_CLAIMS:
            if phrase in low:
                offenders.append(f"{phrase!r} in: {line.strip()[:70]}")

    assert not offenders, (
        "the README claims more than the code supports: " + "; ".join(offenders))


def _normalise(text: str) -> str:
    """Lowercase, with runs of whitespace collapsed to single spaces.

    A README is prose, and prose wraps. Checking for a phrase that happens to
    straddle a line break, or insisting the count be a numeral when the author
    wrote the word, makes a guard that fails on reformatting -- and a guard like
    that gets deleted rather than obeyed.
    """
    return re.sub(r"\s+", " ", text.lower())


def test_the_readme_states_the_agent_count_the_code_constructs():
    section = _society_section()
    flat = _normalise(section)
    count = len(_code_agents())

    words = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
    spelled = words.get(count, str(count))
    assert re.search(rf"\b(?:{count}|{spelled})\s+agents\b", flat), (
        f"the Society section does not state that there are {count} agents; a "
        f"reader has no way to learn the count from the document")

    for agent in sorted(_code_agents()):
        assert re.search(rf"\b{agent}\b", flat), (
            f"the Society section does not name the {agent!r} agent, so the "
            f"stated count is not backed by a list a reader can check")


def test_the_readme_discloses_that_reflection_is_a_stub():
    """A declared stub presented as a working step is the failure to prevent."""
    flat = _normalise(_society_section())

    assert "no live reflection executor" in flat, (
        "the Society section must disclose that the Critic's reflection has no "
        "live executor -- that string is the reason the code gives")
    assert "stub" in flat or "never executed" in flat, (
        "the section should say plainly that the reflection / validate steps are "
        "not measurements")


def test_the_readme_does_not_promise_publication_from_a_blocked_run():
    """`publication_eligible: false` on every run measured here."""
    flat = _normalise(_society_section())
    assert "publication_eligible" in flat, (
        "the Society section should state the eligibility flags a run actually "
        "reports, since a blocked run publishes nothing")
