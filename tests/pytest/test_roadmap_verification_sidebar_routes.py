"""Sidebar <-> route-registry coverage, asserted by script rather than by reading.

Claim under test (L4a #4): "every sidebar item id maps to a real route and
navigates", verified by extracting the sidebar ``items`` arrays and the
``ROUTES`` ids/aliases and asserting **set equality in both directions** -- no
dead nav entry, no orphan route.

The two directions are asked separately because they have different answers, and
the answer depends on *which* sidebar is meant:

* **Direction 1 (no dead nav entry)** -- ``frontend/src/components/Sidebar.vue``
  against ``getRouteById``'s resolution domain (ids + aliases). Holds.
* **Direction 2 (no orphan route)** -- both against ``App.vue`` (the mounted nav)
  and against ``Sidebar.vue``, every registered route is rendered. Set equality
  in both directions holds cleanly.

Every test here is green.
"""

from __future__ import annotations

import re
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src"
SIDEBAR = FRONTEND / "components" / "Sidebar.vue"
APP = FRONTEND / "App.vue"
REGISTRY = FRONTEND / "desktop" / "routeRegistry.ts"


def _read(path: Path) -> str:
    assert path.exists(), f"expected source file missing: {path}"
    return path.read_text(encoding="utf-8")


def _registry_body() -> str:
    text = _read(REGISTRY)
    start = text.index("export const ROUTES")
    return text[start:text.index("\n];", start)]


def sidebar_sections() -> dict:
    """``{section title: [item id, ...]}`` parsed out of ``Sidebar.vue``."""
    text = _read(SIDEBAR)
    start = text.index("const sections = [") + len("const sections = [")
    body = text[start:text.index("\n];", start)]
    sections = {}
    for block in re.finditer(
            r"\{\s*title:\s*'([^']+)'\s*,\s*items:\s*\[([^\]]*)\]",
            body, re.DOTALL):
        ids = re.findall(r"'([^']+)'", block.group(2))
        assert ids, f"parsed no items for section {block.group(1)!r}"
        sections[block.group(1)] = ids
    assert sections, "parsed no sidebar sections; has the layout changed?"
    return sections


def sidebar_item_ids() -> list:
    ids = []
    for section_ids in sidebar_sections().values():
        ids.extend(section_ids)
    return ids


def sidebar_labels() -> set:
    """Keys of ``Sidebar.vue``'s local label map."""
    text = _read(SIDEBAR)
    start = text.index("const labels: Record<string, string> = {")
    body = text[start:text.index("\n};", start)]
    return set(re.findall(r"^\s*([A-Za-z_][\w]*)\s*:", body, re.MULTILINE)) | \
        set(re.findall(r"([A-Za-z_][\w]*):\s*'", body))


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

def test_sidebar_parser_covers_every_declared_section():
    declared = re.findall(r"\{\s*title:\s*'([^']+)'", _read(SIDEBAR))
    assert len(declared) >= 4, declared
    assert set(sidebar_sections()) == set(declared), (
        f"parser missed a section: {set(declared) - set(sidebar_sections())}")


def test_every_sidebar_item_id_resolves_to_a_registered_route():
    ids = sidebar_item_ids()
    missing = sorted(set(ids) - resolvable())
    assert not missing, (
        f"the sidebar renders {len(missing)} id(s) that getRouteById cannot "
        f"resolve, so clicking them navigates nowhere: {missing}")


def test_every_sidebar_item_id_has_a_rendered_label():
    """An id missing from the local label map renders as the raw id."""
    ids = set(sidebar_item_ids())
    unlabelled = sorted(ids - sidebar_labels())
    assert not unlabelled, (
        f"the sidebar renders these ids with no label (raw slug shown): "
        f"{unlabelled}")


def test_sidebar_item_ids_are_unique_and_are_not_alias_slugs():
    ids = sidebar_item_ids()
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicated, f"the same id is rendered twice: {duplicated}"
    offenders = sorted(set(ids) & set(registry_aliases()))
    assert not offenders, (
        f"the sidebar renders alias slugs instead of canonical ids: {offenders}")


def test_registry_ids_and_aliases_do_not_collide():
    clash = sorted(set(registry_route_ids()) & set(registry_aliases()))
    assert not clash, f"an alias slug collides with a route id: {clash}"


# ------------------------------------------------------------------ #
# Direction 2: no orphan route, against the mounted navigation
# ------------------------------------------------------------------ #

def test_app_nav_is_derived_from_the_registry():
    assert app_derives_nav_from_routes(), (
        "App.vue no longer derives navGroups from ROUTES, so its nav may now be "
        "hardcoded and this suite's orphan check would be mirroring a list "
        "instead of reading the registry")


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


def test_legacy_sidebar_is_not_the_mounted_navigation():
    """Records why direction 2 is asserted against App.vue, not Sidebar.vue."""
    importers = [p for p in FRONTEND.rglob("*.vue")
                 if p.name != "Sidebar.vue"
                 and "Sidebar.vue" in p.read_text(encoding="utf-8")]
    importers += [p for p in FRONTEND.rglob("*.ts")
                  if "Sidebar.vue" in p.read_text(encoding="utf-8")]
    assert not importers, (
        "Sidebar.vue is now imported, so it IS mounted navigation; re-point the "
        f"orphan-route assertion at it. Importers: {importers}")


# ------------------------------------------------------------------ #
# Set equality in both directions
# ------------------------------------------------------------------ #

def test_sidebar_and_registry_are_set_equal():
    """Assert set equality in both directions: no dead nav entry, no orphan route.

    ``Sidebar.vue`` renders all 34 ids declared in ``ROUTES``; every one resolves
    via ``getRouteById``, and no route is omitted.
    """
    sidebar = set(sidebar_item_ids())
    registered = set(registry_route_ids())
    omitted = sorted(registered - sidebar)
    assert sidebar == registered, (
        f"Sidebar.vue's ids and routeRegistry's ids are not equal: "
        f"{len(omitted)} registered route(s) appear in no sidebar section: "
        f"{omitted}; {sorted(sidebar - registered)} sidebar id(s) are not route "
        f"ids at all")


def test_the_parsers_are_not_vacuous():
    assert len(registry_route_ids()) >= 30, len(registry_route_ids())
    assert len(sidebar_item_ids()) >= 25, len(sidebar_item_ids())
    assert len(registry_groups()) >= 30, len(registry_groups())
    assert len(registry_aliases()) >= 10, len(registry_aliases())
