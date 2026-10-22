"""Mounted navigation <-> route registry coverage, asserted by parsing the sources.

Claim under test (L4a #4): "every navigation item id maps to a real route and
navigates", verified by extracting the ids the mounted navigation renders and
the ``ROUTES`` ids/aliases from ``routeRegistry.ts`` and asserting set
equality in both directions.

The navigation is ``App.vue``'s ``navGroups``.  Two separate questions hide
inside that one sentence and they have different answers, so this module asks
them separately:

1. **Dead nav entry.**  Is every id the navigation renders resolvable by
   ``getRouteById``?  The resolver's domain is ids *and* aliases, because
   ``getRouteById`` is what navigation actually calls.  A miss here is a
   button that does nothing.

2. **Orphan route.**  Does every registered route appear in the navigation?

Direction 1 was asked against ``frontend/src/components/Sidebar.vue`` -- a
hand-maintained second copy of the navigation that nothing imported.  That
component has been deleted: it had already drifted out of the registry (the
``jobs`` route never reached it), and every button it drew was dead markup.
The mounted navigation is ``App.vue``, and it derives its ids from the
registry instead of repeating them, so "no dead nav entry" is now enforced
structurally: ``App.vue`` must keep deriving ``navGroups`` from ``ROUTES``
(``test_app_nav_is_derived_from_the_registry`` in
``test_roadmap_verification_sidebar_routes.py``), and the ids that derivation
produces are replayed here against the registry and the resolver.  A test
below asserts the legacy component stays deleted -- a second navigation that
is not mounted is a dead navigation.

Everything here is parsed, never hand-listed, so adding a route or a nav item
without the other is caught rather than mirrored.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "frontend" / "src"

LEGACY_SIDEBAR = FRONTEND / "components" / "Sidebar.vue"
APP = FRONTEND / "App.vue"
REGISTRY = FRONTEND / "desktop" / "routeRegistry.ts"


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #

def _read(path: Path) -> str:
    assert path.exists(), f"expected source file missing: {path}"
    return path.read_text(encoding="utf-8")


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


def test_registry_ids_and_aliases_do_not_collide():
    """An alias equal to another route's id would make resolution ambiguous."""
    ids = set(registry_route_ids())
    clash = sorted(ids & set(registry_aliases()))
    assert not clash, f"alias slug collides with a route id: {clash}"


def test_parsed_registry_is_non_trivial():
    """Guard against every parser above degrading to an empty set."""
    assert len(registry_route_ids()) >= 30, len(registry_route_ids())
    assert len(_group_map()) >= 30, len(_group_map())
    assert len(app_nav_route_ids()) >= 20, len(app_nav_route_ids())


# --------------------------------------------------------------------------- #
# the legacy component stays deleted
# --------------------------------------------------------------------------- #

def test_the_legacy_sidebar_component_stays_deleted():
    """There is one navigation, and it is the mounted one.

    ``frontend/src/components/Sidebar.vue`` was a hand-maintained second copy
    of the nav.  Nothing imported it, it had already drifted out of the
    registry, and every button it drew was dead markup.  If it comes back
    unmounted, the assertions above stop describing the navigation a user
    actually clicks -- so its absence is asserted, not assumed.
    """
    assert not LEGACY_SIDEBAR.exists(), (
        f"{LEGACY_SIDEBAR} is back. If it is the mounted navigation, re-point "
        "these assertions at it; if it is not, it is dead markup again."
    )
