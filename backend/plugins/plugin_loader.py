"""Plugin Loader — Dynamic discovery and import of MECH research plugins.

Scans a plugins directory for subpackages that expose a `register()` function
returning a MechPlugin instance. Plugins can also be loaded from arbitrary
Python module paths at runtime.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import pkgutil
import sys
from pathlib import Path
from typing import List, Optional, Type

from .plugin_base import MechPlugin

logger = logging.getLogger(__name__)

_PLUGIN_PACKAGE_ROOT = Path(__file__).parent / "library"


class PluginLoadError(Exception):
    """Raised when a plugin cannot be loaded or does not conform to the SDK."""


class PluginLoader:
    """
    Discovers and imports MechPlugin subclasses from two sources:

    1. **Built-in library** — subpackages under ``backend/plugins/library/``,
       each exposing a top-level ``register()`` → MechPlugin callable.
    2. **External path** — any arbitrary importable Python module path
       supplied by the user at runtime via :meth:`load_from_path`.
    """

    def __init__(self, library_root: Optional[Path] = None) -> None:
        self._library_root = library_root or _PLUGIN_PACKAGE_ROOT
        self._library_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Discovery                                                            #
    # ------------------------------------------------------------------ #

    def discover_builtin_plugins(self) -> List[MechPlugin]:
        """Scan the built-in library directory and return instantiated plugins."""
        plugins: List[MechPlugin] = []

        if not self._library_root.exists():
            return plugins

        # Add parent to path so `backend.plugins.library.*` resolves
        parent = str(self._library_root.parent.parent.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

        for finder, pkg_name, is_pkg in pkgutil.iter_modules([str(self._library_root)]):
            if not is_pkg:
                continue
            module_path = f"backend.plugins.library.{pkg_name}"
            try:
                module = importlib.import_module(module_path)
                plugin = self._extract_plugin(module, module_path)
                if plugin:
                    plugins.append(plugin)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load built-in plugin '%s': %s", pkg_name, exc)

        return plugins

    #: Only modules inside the MECH plugin SDK may be imported by this path.
    #: ``load_from_path`` executes third-party code, so it is not a general
    #: "import anything on sys.path" primitive.
    ALLOWED_MODULE_PREFIX = "backend.plugins.library."

    def load_from_path(self, dotted_module_path: str) -> MechPlugin:
        """Import a bundled library plugin by dotted module path.

        Restricted to ``backend.plugins.library.*``. A caller-supplied path
        previously reached ``importlib.import_module`` with no validation, which
        meant any importable module on ``sys.path`` — including a file an
        attacker had dropped anywhere on the path — would be executed.

        Args:
            dotted_module_path: e.g. ``backend.plugins.library.ioi_experiment_logger``

        Returns:
            Instantiated MechPlugin.

        Raises:
            PluginLoadError: If the module is outside the allowlist, cannot be
                found, or is malformed.
        """
        if not dotted_module_path.startswith(self.ALLOWED_MODULE_PREFIX):
            raise PluginLoadError(
                f"Module '{dotted_module_path}' is outside the plugin library; "
                f"only {self.ALLOWED_MODULE_PREFIX}* may be loaded this way."
            )
        try:
            module = importlib.import_module(dotted_module_path)
        except ModuleNotFoundError as exc:
            raise PluginLoadError(
                f"Module '{dotted_module_path}' not found: {exc}"
            ) from exc

        plugin = self._extract_plugin(module, dotted_module_path)
        if plugin is None:
            raise PluginLoadError(
                f"Module '{dotted_module_path}' has no register() → MechPlugin entry point."
            )
        return plugin

    def load_from_file_sandboxed(self, filepath: str | Path) -> MechPlugin:
        """Execute a plugin file under the sandbox, then bind it via register().

        ``load_from_file`` uses ``importlib`` and therefore executes the file
        with the backend process's full privileges, which makes the install-time
        AST scan the *only* gate. This variant re-runs the AST gate and then
        executes under a restricted global namespace, so a bypass that slipped
        past installation still has no ``__builtins__`` to work with.

        Args:
            filepath: Absolute path to a single plugin .py file.

        Returns:
            Instantiated MechPlugin.

        Raises:
            PluginLoadError: If the file is missing, fails the AST gate, or
                does not yield a MechPlugin from register().
        """
        from .plugin_sandbox import PluginSandbox, SecurityViolation

        filepath = Path(filepath)
        if not filepath.is_file():
            raise PluginLoadError(f"Plugin file not found: {filepath}")

        try:
            source = filepath.read_text(encoding="utf-8")
        except OSError as exc:
            raise PluginLoadError(
                f"Plugin source is unreadable: {filepath} ({exc})") from exc

        try:
            # __file__ is supplied so a plugin can resolve data files next to
            # its own source. The sandbox forbids referencing it dynamically
            # (it is not in FORBIDDEN_ATTRS), but the name must exist or any
            # plugin that uses it raises NameError at import time.
            namespace = PluginSandbox.execute(source, {"__file__": str(filepath)})
        except SecurityViolation as exc:
            raise PluginLoadError(
                f"Plugin '{filepath}' rejected by sandbox: {exc}") from exc

        register_fn = namespace.get("register")
        if register_fn is None or not callable(register_fn):
            raise PluginLoadError(
                f"Plugin file '{filepath}' has no register() → MechPlugin entry point."
            )

        try:
            instance = register_fn()
        except Exception as exc:  # noqa: BLE001
            raise PluginLoadError(
                f"register() in '{filepath}' raised: {exc}") from exc

        if not isinstance(instance, MechPlugin):
            raise PluginLoadError(
                f"register() in '{filepath}' returned {type(instance)!r}, "
                f"expected a MechPlugin subclass."
            )
        return instance

    def load_from_file(self, filepath: str | Path) -> MechPlugin:
        """Import a plugin from an absolute .py file path (no package required).

        Args:
            filepath: Absolute path to the plugin's __init__.py or single .py file.

        Returns:
            Instantiated MechPlugin.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise PluginLoadError(f"Plugin file not found: {filepath}")

        module_name = f"_mech_plugin_{filepath.stem}_{id(filepath)}"
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if spec is None or spec.loader is None:
            raise PluginLoadError(f"Cannot create module spec from: {filepath}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            raise PluginLoadError(f"Error executing plugin file '{filepath}': {exc}") from exc

        plugin = self._extract_plugin(module, str(filepath))
        if plugin is None:
            raise PluginLoadError(
                f"Plugin file '{filepath}' has no register() → MechPlugin entry point."
            )
        return plugin

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_plugin(module, source_label: str) -> Optional[MechPlugin]:
        """Attempt to retrieve a MechPlugin from a module's register() function."""
        register_fn = getattr(module, "register", None)
        if register_fn is None or not callable(register_fn):
            return None

        try:
            instance = register_fn()
        except Exception as exc:  # noqa: BLE001
            raise PluginLoadError(
                f"register() in '{source_label}' raised: {exc}"
            ) from exc

        if not isinstance(instance, MechPlugin):
            raise PluginLoadError(
                f"register() in '{source_label}' returned {type(instance)!r}, "
                f"expected a MechPlugin subclass."
            )
        return instance
