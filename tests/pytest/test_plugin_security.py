"""Plugin subsystem security tests.

Covers the install-time AST lint bypasses, the ``entry`` path traversal in
``_read_source_dir``, and the silent-overwrite hole in ``install_plugin``.

These are regression tests: each one fails against the unpatched tree.
"""
import json
import os
import textwrap

import pytest

from backend.plugins.plugin_sandbox import PluginSandbox, SecurityViolation


# ---------------------------------------------------------------------------
# AST lint â€” bypasses that yielded arbitrary code execution
# ---------------------------------------------------------------------------

BYPASSES = {
    # __import__ is not on the blocked-name list, so this reached os.system.
    "dunder_import": "m = __import__('os')\nm.system('echo pwned')\n",
    # analyze_ast only read node.names[0].name, so a second alias slipped past.
    "second_import_alias": "import json, os\nos.getcwd()\n",
    # node.module is None for relative imports -> `continue` skipped the allowlist.
    "relative_import": "from . import subprocess\nsubprocess.run(['id'])\n",
    "relative_import_from": "from .evil import payload\n",
}


@pytest.mark.parametrize("label", sorted(BYPASSES))
def test_lint_rejects_bypass(label):
    with pytest.raises(SecurityViolation):
        PluginSandbox.analyze_ast(BYPASSES[label])


SANDBOX_ESCAPES = {
    # type + the class MRO is the canonical route to subprocess.Popen.
    "subclasses": "().__class__.__bases__[0].__subclasses__()\n",
    "globals": "print(globals())\n",
    "builtins_attr": "print(__builtins__)\n",
    "getattr_dunder": "getattr((), '__class__')\n",
}


@pytest.mark.parametrize("label", sorted(SANDBOX_ESCAPES))
def test_lint_rejects_dunder_escape(label):
    with pytest.raises(SecurityViolation):
        PluginSandbox.analyze_ast(SANDBOX_ESCAPES[label])


def test_lint_still_permits_sdk_and_safe_stdlib():
    """The fix must not break the allowlist the SDK depends on."""
    ok = textwrap.dedent(
        """
        from __future__ import annotations
        import json
        from typing import Any, Dict
        import math
        from backend.plugins.plugin_base import MechPlugin

        def register() -> MechPlugin:
            return MechPlugin()
        """
    )
    PluginSandbox.analyze_ast(ok)  # must not raise


def test_type_builtin_removed():
    """`type` is the MRO-walking primitive; it must not be handed to plugins."""
    assert "type" not in PluginSandbox.ALLOWED_BUILTINS


# ---------------------------------------------------------------------------
# _read_source_dir â€” `entry` path traversal
# ---------------------------------------------------------------------------


def _write_plugin_dir(tmp_path, name, entry=None):
    directory = tmp_path / name
    directory.mkdir()
    manifest = {"name": "eviltest", "version": "1.0.0"}
    if entry is not None:
        manifest["entry"] = entry
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "good.py").write_text("def register():\n    return None\n", encoding="utf-8")
    return directory


def test_entry_traversal_is_rejected(tmp_path):
    from backend.plugins.service import _read_source_dir

    secret = tmp_path / "secret.py"
    secret.write_text("TOKEN = 'do-not-read'\ndef register(): pass\n", encoding="utf-8")

    directory = _write_plugin_dir(tmp_path, "pkg", entry="../secret.py")

    with pytest.raises(ValueError, match="entry"):
        _read_source_dir(directory)


def test_entry_absolute_path_is_rejected(tmp_path):
    from backend.plugins.service import _read_source_dir

    secret = tmp_path / "secret.py"
    secret.write_text("def register(): pass\n", encoding="utf-8")

    directory = _write_plugin_dir(tmp_path, "pkg2", entry=str(secret))
    with pytest.raises(ValueError, match="entry"):
        _read_source_dir(directory)


def test_entry_inside_directory_still_works(tmp_path):
    from backend.plugins.service import _read_source_dir

    directory = _write_plugin_dir(tmp_path, "pkg3", entry="good.py")
    manifest, source = _read_source_dir(directory)
    assert manifest["name"] == "eviltest"
    assert "def register" in source


# ---------------------------------------------------------------------------
# install_plugin â€” silent overwrite
# ---------------------------------------------------------------------------


def test_install_refuses_silent_overwrite(tmp_path):
    from backend.plugins.plugin_manager import PluginManager

    manager = PluginManager(plugins_dir=str(tmp_path / "installed"))
    manifest = {"name": "squat", "version": "1.0.0"}
    code = "def register():\n    return None\n"

    manager.install_plugin(dict(manifest), code)
    with pytest.raises(ValueError, match="already installed"):
        manager.install_plugin(dict(manifest), "def register():\n    return 'evil'\n")


def test_install_name_cannot_traverse(tmp_path):
    """Defence in depth: a hostile name must not escape plugins_dir."""
    from backend.plugins.plugin_manager import PluginManager

    manager = PluginManager(plugins_dir=str(tmp_path / "installed"))
    with pytest.raises(ValueError):
        manager.install_plugin({"name": "../escaped", "version": "1.0.0"},
                               "def register():\n    return None\n")
    assert not (tmp_path / "escaped.py").exists()


# ---------------------------------------------------------------------------
# _resolve_source â€” install sources must be trusted roots
# ---------------------------------------------------------------------------


def test_untrusted_directory_rejected(tmp_path):
    from backend.plugins.service import _resolve_source

    rogue = tmp_path / "rogue"
    rogue.mkdir()
    with pytest.raises(ValueError, match="not a trusted plugin source"):
        _resolve_source(str(rogue))


def test_trusted_root_allowed(tmp_path, monkeypatch):
    from backend.plugins.service import _resolve_source

    trusted = tmp_path / "trusted"
    plugin_dir = trusted / "myplugin"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setenv("MECH_PLUGIN_TRUSTED_ROOTS", str(trusted))
    assert _resolve_source(str(plugin_dir)) == plugin_dir.resolve()


def test_trusted_root_still_rejects_sibling(tmp_path, monkeypatch):
    """A trusted root must not permit a sibling directory to be installed."""
    from backend.plugins.service import _resolve_source

    trusted = tmp_path / "trusted"
    trusted.mkdir()
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    monkeypatch.setenv("MECH_PLUGIN_TRUSTED_ROOTS", str(trusted))
    with pytest.raises(ValueError, match="not a trusted plugin source"):
        _resolve_source(str(sibling))


def test_sample_alias_still_resolves():
    from backend.plugins.service import _resolve_source

    directory = _resolve_source("sample:ioi_experiment_logger")
    assert directory.is_dir()


def test_sample_alias_traversal_rejected():
    from backend.plugins.service import _resolve_source

    with pytest.raises(ValueError, match="Unknown sample plugin"):
        _resolve_source("sample:../../..")


def test_remote_source_rejected():
    from backend.plugins.service import _resolve_source

    with pytest.raises(ValueError, match="Remote installs are not supported"):
        _resolve_source("https://example.com/plugin.zip")


# ---------------------------------------------------------------------------
# load_from_path â€” must not import arbitrary modules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", [
    "os",
    "subprocess",
    "json",
    "backend.api.dispatcher",
    "",
])
def test_load_from_path_rejects_outside_library(module):
    from backend.plugins.plugin_loader import PluginLoader, PluginLoadError

    with pytest.raises(PluginLoadError, match="outside the plugin library"):
        PluginLoader().load_from_path(module)


def test_load_from_path_allows_library_plugin():
    from backend.plugins.plugin_loader import PluginLoader

    plugin = PluginLoader().load_from_path(
        "backend.plugins.library.ioi_experiment_logger")
    assert plugin.manifest.plugin_id == "mech.example.ioi_experiment_logger"


# ---------------------------------------------------------------------------
# Runtime path â€” plugin code must not run in the backend process
# ---------------------------------------------------------------------------


def test_service_enable_does_not_execute_plugin_in_process():
    """enable() must not run plugin code in the backend process.

    The unpatched tree called PluginLoader.load_from_file -> exec_module, which
    executes with full interpreter privileges. Plugins now run in a bounded
    worker process via start_remote_plugin, so _load_enabled must not resolve
    to any in-process loader.
    """
    import inspect

    from backend.plugins import service as service_mod

    source = inspect.getsource(service_mod.PluginService._load_enabled)
    # Assert on call forms (trailing paren) so a name that merely contains the
    # in-process loader's name is not a false positive.
    for forbidden in (".load_from_file(", ".load_from_file_sandboxed("):
        assert forbidden not in source, (
            f"_load_enabled must not execute plugin code in-process via {forbidden}"
        )
    assert "start_remote_plugin" in source, (
        "_load_enabled must launch a bounded worker process"
    )


def test_sandboxed_loader_rejects_bypass(tmp_path):
    """End-to-end: a payload that reaches os must fail at load time."""
    from backend.plugins.plugin_loader import PluginLoader, PluginLoadError

    payload = tmp_path / "evil.py"
    payload.write_text(
        "import os\n"
        "def register():\n"
        "    os.getcwd()\n"
        "    return None\n",
        encoding="utf-8",
    )

    with pytest.raises(PluginLoadError, match="sandbox"):
        PluginLoader().load_from_file_sandboxed(payload)


def test_sandboxed_loader_accepts_wellformed_plugin(tmp_path):
    """The sandbox must not reject a legitimate plugin."""
    from backend.plugins.plugin_base import MechPlugin
    from backend.plugins.plugin_loader import PluginLoader

    plugin = tmp_path / "good_plugin.py"
    plugin.write_text(
        "from typing import Any, Dict, List, Optional\n"
        "from backend.plugins.plugin_base import MechPlugin, PluginManifest\n"
        "\n"
        "class Demo(MechPlugin):\n"
        "    @property\n"
        "    def manifest(self):\n"
        "        return PluginManifest(\n"
        "            plugin_id='demo', name='Demo', version='1.0.0',\n"
        "            author='t', description='d',\n"
        "        )\n"
        "\n"
        "def register():\n"
        "    return Demo()\n",
        encoding="utf-8",
    )

    assert isinstance(PluginLoader().load_from_file_sandboxed(plugin), MechPlugin)


def test_bundled_sample_plugin_still_loads():
    """The shipped library plugin must survive the tightened lint."""
    from backend.plugins.plugin_loader import PluginLoader
    from backend.plugins.service import _resolve_source

    directory = _resolve_source("sample:ioi_experiment_logger")
    sources = sorted(directory.glob("*.py"))
    assert sources, "sample plugin should contain a .py source"
    plugin = PluginLoader().load_from_file_sandboxed(sources[0])
    assert plugin.manifest.plugin_id


# ---------------------------------------------------------------------------
# Module-alias pivots -- `collections._sys` and friends
#
# Every allowlisted stdlib module holds live module objects as plain
# attributes. Reading one handed the plugin the whole interpreter:
# `sys.modules` reaches any module the host has imported -- including
# `backend.core.auth`, whose `get_bearer_token()` returns the live
# control-plane token -- and `logging.os.environ` is every environment
# variable in the process. The names are not dunders, so a dunder rule could
# not see them.
#
# The aliases below were measured on this host against the modules that are
# actually imported, not assumed from the stdlib layout.
# ---------------------------------------------------------------------------

ESCAPE_ALIASES = {
    "collections._sys": "import collections\nx = collections._sys\n",
    "logging.sys": "import logging\nx = logging.sys\n",
    "logging.os": "import logging\nx = logging.os\n",
    "enum.sys": "import enum\nx = enum.sys\n",
    "enum.bltns": "import enum\nx = enum.bltns\n",
}

# The same pivots as (module, attribute), for driving the runtime guard
# directly with the lint gate deliberately out of the way.
ESCAPE_ATTRS = {
    "collections._sys": ("collections", "_sys"),
    "logging.sys": ("logging", "sys"),
    "logging.os": ("logging", "os"),
    "enum.sys": ("enum", "sys"),
    "enum.bltns": ("enum", "bltns"),
}


@pytest.mark.parametrize("label", sorted(ESCAPE_ALIASES))
def test_module_alias_pivot_rejected_by_lint(label):
    with pytest.raises(SecurityViolation):
        PluginSandbox.analyze_ast(ESCAPE_ALIASES[label])


@pytest.mark.parametrize("label", sorted(ESCAPE_ATTRS))
def test_runtime_guard_refuses_escape_attributes(label):
    """The runtime layer must hold on its own, with no lint gate involved.

    This calls the guarded import directly: `analyze_ast` is never consulted,
    so a regression that removed only the static rule would still fail here.
    """
    module_name, attr = ESCAPE_ATTRS[label]
    guarded = PluginSandbox._guarded_import(module_name)  # noqa: SLF001
    with pytest.raises(SecurityViolation):
        getattr(guarded, attr)


def test_import_from_alias_pivot_rejected():
    """`from collections import _sys` produces no Attribute node at all."""
    source = "from collections import _sys\nx = _sys\n"
    with pytest.raises(SecurityViolation):
        PluginSandbox.analyze_ast(source)
    with pytest.raises(SecurityViolation):
        PluginSandbox.execute(source)


def test_bearer_token_module_is_not_reachable_from_plugin_code():
    """The F-1 proof of concept, asserted with the route actually live.

    `backend.core.auth` is imported first on purpose. Without it the lookup
    raises KeyError on the unpatched tree as well, which made an earlier
    version of this test pass for the wrong reason. With it imported, the
    unpatched tree really reaches the module and the patched tree is refused at
    `collections._sys`. The token itself is never read or echoed, so a
    regression cannot leak the thing this test protects.
    """
    import backend.core.auth  # noqa: F401  -- makes the route reachable

    with pytest.raises(SecurityViolation):
        PluginSandbox.execute(
            "import collections\n"
            "auth = collections._sys.modules['backend.core.auth']\n"
        )


def test_environment_is_not_reachable_from_plugin_code():
    """The F-2 proof of concept, asserted closed end to end."""
    with pytest.raises(SecurityViolation):
        PluginSandbox.execute(
            "import logging\n"
            "leaked = dict(logging.os.environ)\n"
        )


def test_nested_module_attribute_is_guarded():
    """A forwarded submodule must not become an unguarded way back in."""
    with pytest.raises(SecurityViolation):
        PluginSandbox.execute(
            "import logging.handlers\nx = logging.handlers.os\n")


def test_guard_does_not_refuse_ordinary_field_names():
    """`self.os` is a field, not the module: refusing it bought nothing.

    The earlier name-level rule refused working code -- `self.io = None`,
    `self.os = 'win32'`, `self.builtins = []` -- while missing the payload it
    was written for. A store binds a field; only a load can be a pivot.
    """
    PluginSandbox.analyze_ast(textwrap.dedent(
        """
        class C:
            def __init__(self):
                self.os = 'win32'
                self.builtins = []
                self.io = None
        """
    ))


def test_forwarded_stdlib_still_works():
    """Forwarding a function leaves it bound to the real module globals.

    `collections.namedtuple` calls `_sys._getframe(1)` internally; closing the
    pivot must not break it, or the fix trades one defect for another.
    """
    namespace = PluginSandbox.execute(textwrap.dedent(
        """
        import collections
        from collections import OrderedDict
        Point = collections.namedtuple('Point', ['x', 'y'])
        point = Point(1, 2)
        ordered = OrderedDict()
        ordered['a'] = 1
        """
    ))
    assert namespace["point"].x == 1
    assert list(namespace["ordered"]) == ["a"]


def test_guarded_module_hides_the_real_module_object():
    """The real module must not be recoverable off the proxy it hands back."""
    namespace = PluginSandbox.execute("import collections\n")
    guarded = namespace["collections"]
    assert guarded.__name__ == "collections"
    assert "_sys" not in guarded.__dict__, (
        "the proxy must not carry the real module's escape attributes")
    with pytest.raises(SecurityViolation):
        guarded._sys  # noqa: B018


# ---------------------------------------------------------------------------
# Aliased imports -- the case the lint rule structurally cannot catch
#
# The AST rule matches an attribute read whose base is a Name in the import
# allowlist, so `import collections as c; c._sys` produces nothing for it to
# match. The runtime proxy is what closes this, and these tests pin that: if the
# proxy regressed, the lint gate would still pass and the escape would return.
# ---------------------------------------------------------------------------

ALIASED_PIVOTS = {
    "aliased_sys": "import collections as c\nx = c._sys\n",
    "aliased_logging_os": "import logging as L\nx = L.os.environ\n",
    "aliased_enum_bltns": "import enum as E\nx = E.bltns\n",
}


@pytest.mark.parametrize("label", sorted(ALIASED_PIVOTS))
def test_the_lint_gate_alone_does_not_catch_an_aliased_pivot(label):
    """Assert the premise, so the test above cannot pass vacuously.

    If a future change made `analyze_ast` catch the alias form, this stops being
    the runtime layer's job -- and this test says so out loud instead of leaving
    the proxy looking load-bearing when it is not.
    """
    PluginSandbox.analyze_ast(ALIASED_PIVOTS[label])  # must not raise


@pytest.mark.parametrize("label", sorted(ALIASED_PIVOTS))
def test_aliased_pivot_is_closed_by_the_runtime_layer(label):
    with pytest.raises(SecurityViolation):
        PluginSandbox.execute(ALIASED_PIVOTS[label])


def test_aliased_legitimate_import_still_works():
    namespace = PluginSandbox.execute(
        "import collections as c\n"
        "Point = c.namedtuple('Point', ['x'])\n"
        "point = Point(1)\n"
    )
    assert namespace["point"].x == 1
