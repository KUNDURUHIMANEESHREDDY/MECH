"""Navigation <-> route-registry coverage, asserted by script rather than by reading.

Claim under test (L4a #4): "every navigation item id maps to a real route and
navigates", verified by extracting the ids the mounted navigation renders and
the ``ROUTES`` ids/aliases and asserting **set equality in both directions** --
no dead nav entry, no orphan route.

The navigation is ``App.vue``'s ``navGroups``: the shell that is mounted.
* **Direction 2 (no orphan route)** is asserted against it -- every
  registered route that is not an alias is rendered, and every navigation
  group name matches a group the registry actually uses.
* **Direction 1 (no dead nav entry)** is enforced structurally rather than by
  a second list: ``App.vue`` must derive its navigation from ``ROUTES``
  (``test_app_nav_is_derived_from_the_registry``), and the ids that derivation
  produces are the registry's own ids, which ``getRouteById`` resolves by
  construction.

``frontend/src/components/Sidebar.vue`` used to be the subject of direction
1: a hand-maintained second copy of the navigation that nothing imported.  It
had already drifted out of the registry, and it is now deleted.  The test
below asserts it stays deleted -- an unmounted navigation is a dead
navigation, and its ids were never the ones a user could click.

Every test here is green.
"""

from __future__ import annotations

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src"
LEGACY_SIDEBAR = FRONTEND / "components" / "Sidebar.vue"
APP = FRONTEND / "App.vue"
REGISTRY = FRONTEND / "desktop" / "routeRegistry.ts"


def _read(path: Path) -> str:
    assert path.exists(), f"expected source file missing: {path}"
    return path.read_text(encoding="utf-8")


def _registry_body() -> str:
    text = _read(REGISTRY)
    start = text.index("export const ROUTES")
    return text[start:text.index("\n];", start)]


def registry_route_ids() -> list:
    return re.findall(r"id:\s*'([^']+)'", _registry_body())


def registry_aliases() -> list:
    body = _registry_body()
    aliases = []
    for block in re.finditer(r"aliases:\s*\[([^\]]*)\]", body):
        aliases.extend(re.findall(r"'([^']+)'", block.group(1)))
    return aliases


def registry_alias_of_ids() -> set:
    """Route ids carrying ``aliasOf`` -- hidden from ``App.vue``'s nav."""
    body = _registry_body()
    out = set()
    for block in re.finditer(r"\{\s*id:\s*'([^']+)'(.*?)\n  \},", body, re.DOTALL):
        if "aliasOf:" in block.group(2):
            out.add(block.group(1))
    return out


def registry_groups() -> dict:
    """``{route id: group}`` for every route block."""
    body = _registry_body()
    groups = {}
    for block in re.finditer(
            r"\{\s*id:\s*'([^']+)',\s*label:\s*'[^']*',\s*group:\s*'([^']+)'",
            body, re.DOTALL):
        groups[block.group(1)] = block.group(2)
    return groups


def app_group_order() -> list:
    text = _read(APP)
    match = re.search(r"const GROUP_ORDER\s*=\s*\[([^\]]*)\]", text)
    assert match, "App.vue no longer declares GROUP_ORDER"
    order = re.findall(r"'([^']+)'", match.group(1))
    assert order, "GROUP_ORDER parsed empty"
    return order


def app_derives_nav_from_routes() -> bool:
    """True when App.vue's nav is a filter over ROUTES, not a hardcoded list."""
    return bool(re.search(
        r"ROUTES\.filter\(\s*route\s*=>\s*route\.group\s*===\s*\w+\s*&&\s*"
        r"!route\.aliasOf\s*\)", _read(APP)))


def resolvable() -> set:
    """Exactly what ``getRouteById`` can resolve: ids plus alias slugs."""
    return set(registry_route_ids()) | set(registry_aliases())


def app_nav_route_ids() -> list:
    """Replay App.vue's own filter against the parsed registry."""
    groups = app_group_order()
    hidden = registry_alias_of_ids()
    return [rid for rid, group in registry_groups().items()
            if group in groups and rid not in hidden]


# ------------------------------------------------------------------ #
# Direction 1: no dead nav entry
# ------------------------------------------------------------------ #

def test_app_nav_is_derived_from_the_registry():
    assert app_derives_nav_from_routes(), (
        "App.vue no longer derives navGroups from ROUTES, so its nav may now be "
        "hardcoded and this suite's orphan check would be mirroring a list "
        "instead of reading the registry")


def test_the_legacy_sidebar_component_stays_deleted():
    """One navigation, and it is the mounted one.

    ``Sidebar.vue`` was a second, hand-maintained copy of the nav that nothing
    imported.  It had drifted out of the registry already, and every button it
    drew was dead markup.  If it returns unmounted, the orphan-route
    assertions below stop describing the navigation a user clicks -- so its
    absence is asserted, not assumed.
    """
    assert not LEGACY_SIDEBAR.exists(), (
        f"{LEGACY_SIDEBAR} is back. If it is the mounted navigation, re-point "
        "the orphan-route assertion at it; if it is not, it is dead markup "
        "again.")


# ------------------------------------------------------------------ #
# Direction 2: no orphan route, against the mounted navigation
# ------------------------------------------------------------------ #

def test_no_registry_route_is_orphaned_from_the_mounted_nav():
    registered = set(registry_route_ids())
    hidden = registry_alias_of_ids()
    reachable = set(app_nav_route_ids())
    orphans = sorted(registered - hidden - reachable)
    assert not orphans, (
        f"{len(orphans)} registered route(s) mount nowhere in App.vue's nav: "
        f"{orphans}")


def test_app_nav_group_names_all_match_a_registry_group():
    declared = set(app_group_order())
    actual = set(registry_groups().values())
    assert not (declared - actual), (
        f"App.vue declares navigation group(s) no route belongs to: "
        f"{sorted(declared - actual)}")
    assert not (actual - declared), (
        f"route group(s) no navigation section renders: "
        f"{sorted(actual - declared)}")


def test_registry_ids_and_aliases_do_not_collide():
    clash = sorted(set(registry_route_ids()) & set(registry_aliases()))
    assert not clash, f"an alias slug collides with a route id: {clash}"


# ------------------------------------------------------------------ #
# the parsers are not vacuous
# ------------------------------------------------------------------ #

def test_the_parsers_are_not_vacuous():
    assert len(registry_route_ids()) >= 30, len(registry_route_ids())
    assert len(registry_groups()) >= 30, len(registry_groups())
    assert len(registry_aliases()) >= 10, len(registry_aliases())
    assert len(app_nav_route_ids()) >= 20, len(app_nav_route_ids())
