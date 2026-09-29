"""HTTP-facing service over the existing MECH plugin SDK.

Wires together the SDK pieces that shipped without any caller:

* :class:`PluginManager` — install (with sandbox AST scan), enable/disable
  metadata, JSON registry persistence;
* :class:`PluginLoader` — load an installed ``.py`` file's ``register()``
  entry point into a live ``MechPlugin``;
* :class:`PluginRegistry` process singleton — live runtime instances;
* :class:`PluginHookBus` — fan-out of platform hook events.

Install sources are local directories containing a ``manifest.json`` plus the
plugin source file, or the ``sample:<name>`` alias resolving to a bundled
example under ``backend/plugins/library/``. There is no remote install path.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .plugin_base import MechPlugin
from .plugin_hooks import PluginHookBus
from .plugin_loader import PluginLoader, PluginLoadError
from .plugin_manager import PluginManager
from .plugin_registry import PluginNotFoundError, PluginRegistry

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,63}$")
_SAMPLE_PREFIX = "sample:"

_lock = threading.RLock()


def _library_dir() -> Path:
    return Path(__file__).resolve().parent / "library"


def _read_source_dir(directory: Path) -> tuple[Dict[str, Any], str]:
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{directory} contains no manifest.json")
    except (OSError, ValueError) as exc:
        raise ValueError(f"manifest.json is unreadable: {exc}")
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be a JSON object")
    for field in ("name", "version"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"manifest.json requires a non-empty '{field}'")
    name = manifest["name"].strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            "manifest 'name' must be 2-64 chars: lowercase, digits, '_', '.', '-'"
        )
    candidates = sorted(
        p for p in directory.glob("*.py")
        if p.is_file() and p.name != "manifest.json"
    )
    base = directory.resolve()
    entry = manifest.get("entry")
    if isinstance(entry, str) and entry.strip():
        # ``entry`` is attacker-controlled manifest data. Resolve it and
        # require containment, so ``../`` or an absolute path cannot read a
        # .py file outside the plugin directory.
        chosen = (base / entry.strip()).resolve()
        if not chosen.is_relative_to(base):
            raise ValueError(
                "manifest 'entry' must name a .py file inside the plugin directory"
            )
        if chosen.suffix != ".py" or not chosen.is_file():
            raise ValueError(f"manifest entry '{entry}' was not found")
    elif len(candidates) == 1:
        chosen = candidates[0]
    elif not candidates:
        raise ValueError("plugin directory contains no Python source file")
    else:
        raise ValueError(
            "plugin directory has several .py files; set manifest 'entry'")
    try:
        source = chosen.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"plugin source is unreadable: {exc}")
    if "def register" not in source:
        raise ValueError("plugin source must define a register() entry point")
    manifest["name"] = name
    return manifest, source


def _resolve_source(source: str) -> Path:
    raw = (source or "").strip()
    if not raw:
        raise ValueError("A local plugin directory path is required.")
    lowered = raw.lower()
    if "://" in lowered or lowered.startswith(("http:", "https:", "ftp:")):
        raise ValueError("Remote installs are not supported; use a local directory.")
    if raw.startswith(_SAMPLE_PREFIX):
        candidate = (_library_dir() / raw[len(_SAMPLE_PREFIX):]).resolve()
        if _library_dir().resolve() not in candidate.parents:
            raise ValueError("Unknown sample plugin.")
        if not candidate.is_dir():
            raise ValueError(f"Sample plugin '{raw}' does not exist.")
        return candidate
    candidate = Path(raw).expanduser().resolve()
    if not candidate.is_dir():
        raise ValueError(f"'{raw}' is not an existing local directory.")
    return candidate


class PluginService:
    """Single entry point used by the HTTP dispatcher."""

    def __init__(self, manager: Optional[PluginManager] = None) -> None:
        self._manager = manager or PluginManager()
        self._loader = PluginLoader()
        self._registry = PluginRegistry.global_instance()
        self._bus: Optional[PluginHookBus] = None

    # -- runtime ------------------------------------------------------
    def bus(self) -> PluginHookBus:
        if self._bus is None:
            self._bus = PluginHookBus(self._registry)
        return self._bus

    def _load_enabled(self, name: str) -> MechPlugin:
        info = self._manager.registry.get(name)
        if info is None:
            raise ValueError(f"No installed plugin named '{name}'.")
        plugin = self._loader.load_from_file_sandboxed(info["path"])
        manifest = plugin.manifest
        # Pin the SDK plugin_id on first load; later loads must present the
        # same id, so a swapped-in file cannot silently take another name.
        expected = info.get("plugin_id")
        if expected:
            if manifest.plugin_id != expected:
                raise PluginLoadError(
                    f"register() returned '{manifest.plugin_id}', "
                    f"expected '{expected}'.")
        else:
            info["plugin_id"] = manifest.plugin_id
            self._manager.save_registry()
        self._registry.register(plugin, allow_override=True)
        return plugin

    def restore_enabled(self) -> List[str]:
        """Load every enabled plugin at startup. Never raises."""
        restored: List[str] = []
        try:
            names = [name for name, info in self._manager.registry.items()
                     if info.get("enabled")]
        except Exception:
            return restored
        for name in names:
            try:
                self._load_enabled(name)
                restored.append(name)
            except Exception:
                continue
        return restored

    # -- catalog ------------------------------------------------------
    def _live_by_plugin_id(self) -> Dict[str, Any]:
        return {plugin.manifest.plugin_id: plugin
                for plugin in self._registry.list_plugins()}

    def list_all(self) -> List[Dict[str, Any]]:
        live_by_id = self._live_by_plugin_id()
        records = []
        for name, info in self._manager.registry.items():
            live = live_by_id.get(str(info.get("plugin_id", "")))
            manifest = {
                "plugin_id": str(info.get("plugin_id", name)),
                "name": name,
                "version": str(info.get("version", "")),
                "author": str(info.get("author", "")),
                "description": str(info.get("description", "")),
                "hooks": [],
                "dependencies": list(info.get("dependencies", [])),
            }
            if live is not None:
                manifest = {
                    "plugin_id": live.manifest.plugin_id,
                    "name": live.manifest.name,
                    "version": live.manifest.version,
                    "author": live.manifest.author,
                    "description": live.manifest.description,
                    "hooks": list(live.manifest.hooks),
                    "dependencies": list(live.manifest.dependencies),
                }
            records.append({
                "id": name,
                "name": manifest["name"],
                "version": manifest["version"],
                "author": manifest["author"],
                "description": manifest["description"],
                "hooks": manifest["hooks"],
                "dependencies": manifest["dependencies"],
                "enabled": bool(info.get("enabled", False)),
                "loaded": live is not None,
                "source": "backend",
            })
        records.sort(key=lambda item: item["name"].lower())
        return records

    # -- lifecycle ----------------------------------------------------
    def install(self, source: str) -> Dict[str, Any]:
        directory = _resolve_source(source)
        manifest, code = _read_source_dir(directory)
        with _lock:
            message = self._manager.install_plugin(dict(manifest), code)
        return {"name": manifest["name"], "message": message,
                "enabled": False}

    def enable(self, name: str) -> Dict[str, Any]:
        with _lock:
            if name not in self._manager.registry:
                raise ValueError(f"No installed plugin named '{name}'.")
            self._manager.enable_plugin(name)
            try:
                self._load_enabled(name)
            except Exception as exc:
                try:
                    self._manager.disable_plugin(name)
                except Exception:
                    pass
                raise ValueError(f"Plugin '{name}' failed to load: {exc}")
        return self.describe(name)

    def _runtime_id(self, name: str) -> Optional[str]:
        info = self._manager.registry.get(name) or {}
        plugin_id = info.get("plugin_id")
        return str(plugin_id) if plugin_id else None

    def _unregister_runtime(self, name: str) -> None:
        plugin_id = self._runtime_id(name)
        if plugin_id:
            try:
                self._registry.unregister(plugin_id)
                return
            except PluginNotFoundError:
                pass
        try:
            self._registry.unregister(name)
        except PluginNotFoundError:
            pass

    def disable(self, name: str) -> Dict[str, Any]:
        with _lock:
            if name not in self._manager.registry:
                raise ValueError(f"No installed plugin named '{name}'.")
            self._unregister_runtime(name)
            self._manager.disable_plugin(name)
        return self.describe(name)

    def uninstall(self, name: str) -> Dict[str, Any]:
        import shutil
        with _lock:
            info = self._manager.registry.get(name)
            if info is None:
                raise ValueError(f"No installed plugin named '{name}'.")
            self._unregister_runtime(name)
            path = info.get("path", "")
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
            del self._manager.registry[name]
            self._manager.save_registry()
        return {"name": name, "uninstalled": True}

    def describe(self, name: str) -> Dict[str, Any]:
        for record in self.list_all():
            if record["id"] == name or record["name"] == name:
                return record
        raise ValueError(f"No installed plugin named '{name}'.")


_service: Optional[PluginService] = None


def get_service() -> PluginService:
    global _service
    if _service is None:
        _service = PluginService()
    return _service
