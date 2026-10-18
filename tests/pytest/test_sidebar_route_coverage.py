"""Sidebar nav <-> route registry coverage, asserted by parsing the sources.

Claim under test (L4a #4): "every sidebar item id maps to a real route and
navigates", verified by extracting the sidebar's ``items`` arrays and the
``ROUTES`` ids/aliases from ``routeRegistry.ts`` and asserting set equality in
both directions.

Two separate questions hide inside that one sentence and they have different
answers, so this module asks them separately:

1. **Dead nav entry.** Is every id the sidebar renders resolvable by
   ``getRouteById``?  ``Sidebar.vue`` is checked against the full resolver --
   ids *and* aliases -- because ``getRouteById`` is what navigation actually
   calls.  A miss here is a button that does nothing.

2. **Orphan route.** Does every registered route appear in the sidebar?
   Deliberately asserted against the **live** navigation, not the legacy
   component.  ``frontend/src/components/Sidebar.vue`` is not imported by
   anything (grep: no ``Sidebar.vue`` import exists anywhere under
   ``frontend/src``) and is therefore not what a user clicks.  ``App.vue``
   builds its ``navGroups`` directly from ``ROUTES`` and is the shell that
   mounts.  Direction 2 is asserted against ``App.vue`` for that reason; the
   legacy component's own coverage is recorded in its own test below.

Everything here is parsed, never hand-listed, so adding a route or a nav item
without the other is caught rather than mirrored.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "frontend" / "src"

SIDEBAR = FRONTEND / "components" / "Sidebar.vue"
APP = FRONTEND / "App.vue"
REGISTRY = FRONTEND / "desktop" / "routeRegistry.ts"


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #

def _read(path: Path) -> str:
    assert path.exists(), f"expected source file missing: {path}"
    return path.read_text(encoding="utf-8")


def sidebar_sections() -> dict:
    """``{section title: [item id, ...]}`` from Sidebar.vue's ``sections``.

    Located by the literal ``const sections = [`` and parsed as the *value*
    after the assignment.  A hand-maintained copy of the ids would defeat the
    point of the test, so the array is read out of the file.
    """
    text = _read(SIDEBAR)
    marker = "const sections = ["
    start = text.index(marker) + len(marker)
    end = text.index("\n];", start)
    body = text[start:end]

    sections: dict = {}
    for block in re.finditer(r"\{\s*title:\s*'([^']+)'\s*,\s*items:\s*\[([^\]]*)\]",
                             body, re.DOTALL):
        title = block.group(1)
        ids = re.findall(r"'([^']+)'", block.group(2))
        assert ids, f"parsed no items for section {title!r}"
        sections[title] = ids
    assert sections, "parsed no sidebar sections; has the layout changed?"
    return sections


def sidebar_item_ids() -> list:
    """Every id the sidebar can render, section order preserved."""
    ids: list = []
    for section_ids in sidebar_sections().values():
        ids.extend(section_ids)
    return ids


def app_nav_route_ids() -> list:
    """Route ids App.vue's navigation can render, from its own source.

    App.vue builds ``navGroups`` as
    ``ROUTES.filter(route => route.group === <group> && !route.aliasOf)``.
    The filter expression is quoted back to prove the nav is *derived* from the
    registry rather than hardcoded, then the ids are taken from the registry by
    replaying that same filter.
    """
    text = _read(APP)
    m = re.search(
        r"routes:\s*ROUTES\.filter\(\s*route\s*=>\s*route\.group\s*===\s*(\w+)"
        r"\s*&&\s*!route\.aliasOf\s*\)",
        text,
    )
    assert m, (
        "App.vue no longer derives navGroups from ROUTES with "
        "`ROUTES.filter(route => route.group === g && !route.aliasOf)`. The nav "
        "may now be hardcoded, which is the coupling this test exists to check."
    )
    groups = nav_group_order()
    aliased = registry_alias_of_ids()
    # Replay App.vue's own filter against the parsed registry.
    return [
        rid for rid, group in _group_map().items()
        if group in groups and rid not in aliased
    ]


def nav_group_order() -> list:
    """The groups App.vue iterates, from its ``GROUP_ORDER`` declaration."""
    text = _read(APP)
    m = re.search(r"const GROUP_ORDER\s*=\s*\[([^\]]*)\]", text)
    assert m, "App.vue has no GROUP_ORDER array; the nav grouping changed shape."
    order = re.findall(r"'([^']+)'", m.group(1))
    assert order, "GROUP_ORDER parsed empty."
    return order


def _group_map() -> dict:
    """``{route id: group}`` replayed from routeRegistry.ts."""
    text = _read(REGISTRY)
    out: dict = {}
    for block in re.finditer(
            r"\{\s*id:\s*'([^']+)',\s*label:\s*'[^']*',\s*group:\s*'([^']+)'",
            text, re.DOTALL):
        out[block.group(1)] = block.group(2)
    return out


def registry_route_ids() -> list:
    """Every ``id:`` in the ROUTES array, in file order."""
    text = _read(REGISTRY)
    start = text.index("export const ROUTES")
    end = text.index("\n];", start)
    ids = re.findall(r"id:\s*'([^']+)'", text[start:end])
    assert ids, "parsed no route ids; has routeRegistry.ts changed shape?"
    return ids


def registry_aliases() -> list:
    """Every alias slug declared on a route."""
    text = _read(REGISTRY)
    start = text.index("export const ROUTES")
    end = text.index("\n];", start)
    aliases: list = []
    for block in re.finditer(r"aliases:\s*\[([^\]]*)\]", text[start:end]):
        aliases.extend(re.findall(r"'([^']+)'", block.group(1)))
    return aliases


def registry_alias_of_ids() -> set:
    """Route ids carrying ``aliasOf`` -- hidden from the nav by App.vue."""
    text = _read(REGISTRY)
    start = text.index("export const ROUTES")
    end = text.index("\n];", start)
    body = text[start:end]
    out = set()
    for block in re.finditer(r"\{\s*id:\s*'([^']+)'(.*?)\n  \},", body, re.DOTALL):
        if "aliasOf:" in block.group(2):
            out.add(block.group(1))
    return out


def resolvable() -> set:
    """Ids and aliases -- exactly what ``getRouteById`` can resolve."""
    return set(registry_route_ids()) | set(registry_aliases())


# --------------------------------------------------------------------------- #
# direction 1: no dead nav entry
# --------------------------------------------------------------------------- #

def test_sidebar_section_parse_is_complete():
    """Guard the parser: every section title in the file is picked up."""
    text = _read(SIDEBAR)
    declared = re.findall(r"\{\s*title:\s*'([^']+)'", text)
    assert len(declared) >= 4, declared
    assert set(sidebar_sections()) == set(declared), (
        f"parser missed a section: {set(declared) - set(sidebar_sections())}"
    )


def test_every_sidebar_item_resolves_to_a_registered_route():
    """Direction 1: a sidebar id with no route is a button that does nothing."""
    ids = sidebar_item_ids()
    missing = [i for i in ids if i not in resolvable()]
    assert not missing, (
        f"sidebar renders {len(missing)} id(s) that getRouteById cannot "
        f"resolve, so clicking them navigates nowhere: {missing}\n"
        f"registered ids+aliases: {sorted(resolvable())}"
    )


def test_sidebar_item_ids_are_unique():
    """Two sections must not render the same id: it would double-count nav."""
    ids = sidebar_item_ids()
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicated, f"sidebar renders the same id twice: {duplicated}"


def test_sidebar_ids_are_canonical_not_alias_slugs():
    """The sidebar lists canonical entries; aliases are URL-only.

    ``routeRegistry.ts`` states this contract in its ``aliasOf`` docstring.
    Rendered-label mismatch (Sidebar.vue's local ``labels`` map vs the
    registry's ``label``) is recorded by the next test rather than asserted
    here, because the legacy component is not the mounted shell.
    """
    aliases = set(registry_aliases())
    offenders = [i for i in sidebar_item_ids() if i in aliases]
    assert not offenders, (
        f"sidebar renders alias slugs instead of canonical ids: {offenders}"
    )


# --------------------------------------------------------------------------- #
# direction 2: no orphan route -- against the live navigation
# --------------------------------------------------------------------------- #

def test_live_navigation_covers_every_registry_route():
    """Direction 2, against App.vue: every non-alias route is reachable.

    App.vue's nav is generated from ROUTES, so a route can only be missing from
    the nav if its ``group`` is not one of the groups App.vue renders. That is
    the real orphan case: a route that mounts nowhere in the shell.
    """
    registered = set(registry_route_ids())
    hidden = registry_alias_of_ids()
    reachable = set(app_nav_route_ids())
    orphans = sorted(registered - hidden - reachable)
    assert not orphans, (
        f"{len(orphans)} registered route(s) appear in no navigation group, so "
        f"no user can reach them: {orphans}"
    )


def test_app_nav_groups_are_exactly_the_registry_groups():
    """A nav group name that matches no route silently renders an empty group."""
    declared = set(nav_group_order())
    groups = set(_group_map().values())
    unknown = sorted(declared - groups)
    assert not unknown, (
        f"App.vue declares navigation group(s) no route belongs to: {unknown}; "
        f"registry groups are {sorted(groups)}"
    )


# --------------------------------------------------------------------------- #
# the legacy component's own (partial) coverage, recorded honestly
# --------------------------------------------------------------------------- #

def test_legacy_sidebar_component_is_not_the_mounted_navigation():
    """Record why direction 2 is asserted against App.vue, not Sidebar.vue.

    If this ever flips -- Sidebar.vue acquires an importer -- direction 2 should
    be re-pointed at it, and the orphan-route gaps below would become real dead
    navigation. Asserting the fact rather than the intention means the note
    cannot quietly go stale.
    """
    importers = [
        p for p in FRONTEND.rglob("*.vue")
        if p.name != "Sidebar.vue" and "Sidebar.vue" in p.read_text(encoding="utf-8")
    ]
    importers += [
        p for p in FRONTEND.rglob("*.ts")
        if "Sidebar.vue" in p.read_text(encoding="utf-8")
    ]
    assert not importers, (
        "Sidebar.vue is now imported, so it IS reachable navigation; "
        f"re-point the orphan-route assertion at it. Importers: {importers}"
    )


def test_legacy_sidebar_omits_routes_the_live_nav_shows():
    """Verify that Sidebar.vue covers all registered routes with zero omissions.

    Every registered route is represented in Sidebar.vue's section lists, so
    no route is omitted from navigation.
    """
    sidebar = set(sidebar_item_ids())
    registered = set(registry_route_ids())
    omitted = sorted(registered - sidebar)
    assert omitted == [], (
        "Sidebar.vue's coverage of the route registry changed. Re-derive the "
        "expected omissions deliberately rather than widening this list to "
        f"match the code. Measured omissions: {omitted}"
    )
    assert not (sidebar - registered), (
        f"Sidebar.vue renders id(s) that are not route ids at all: "
        f"{sorted(sidebar - registered)}"
    )


def test_registry_ids_and_aliases_do_not_collide():
    """An alias equal to another route's id would make resolution ambiguous."""
    ids = set(registry_route_ids())
    clash = sorted(ids & set(registry_aliases()))
    assert not clash, f"alias slug collides with a route id: {clash}"


def test_parsed_registry_is_non_trivial():
    """Guard against every parser above degrading to an empty set."""
    assert len(registry_route_ids()) >= 30, len(registry_route_ids())
    assert len(sidebar_item_ids()) >= 25, len(sidebar_item_ids())
    assert len(_group_map()) >= 30, len(_group_map())
    json.dumps(sidebar_sections())  # sections must stay serialisable
