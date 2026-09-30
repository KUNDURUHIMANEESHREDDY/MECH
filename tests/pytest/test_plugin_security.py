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
# AST lint — bypasses that yielded arbitrary code execution
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
# _read_source_dir — `entry` path traversal
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
# install_plugin — silent overwrite
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
# _resolve_source — install sources must be trusted roots
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
# load_from_path — must not import arbitrary modules
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
# Runtime path — plugin code must not run in the backend process
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
