import os
import json
import re
from typing import Dict, Any, List
from .plugin_dependency import PluginDependencyResolver
from .plugin_sandbox import PluginSandbox

class PluginManager:
    """
    Manages installation, versioning, enabling, and dependency resolution of plugins.
    """
    def __init__(self, plugins_dir: str = "backend/plugins/installed"):
        self.plugins_dir = plugins_dir
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.active_plugins: List[str] = []
        os.makedirs(self.plugins_dir, exist_ok=True)
        self.load_registry()

    def load_registry(self):
        registry_path = os.path.join(self.plugins_dir, "registry.json")
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                self.registry = json.load(f)

    def save_registry(self):
        registry_path = os.path.join(self.plugins_dir, "registry.json")
        with open(registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def install_plugin(self, manifest: Dict[str, Any], source_code: str):
        """
        Installs a plugin given its manifest and source code.
        Manifest should contain: name, version, dependencies [list], description.
        """
        name = manifest["name"]
        version = manifest["version"]

        # Defence in depth: the service layer already enforces this, but
        # install_plugin is a public entry point and must not trust its
        # caller to have validated the name. A name containing a separator
        # would write outside plugins_dir.
        if not re.match(r"^[a-z0-9][a-z0-9_.-]{1,63}$", name):
            raise ValueError(
                "Plugin name must be 2-64 chars: lowercase, digits, '_', '.', '-'"
            )

        # Refuse to overwrite an installed plugin. Overwriting in place let a
        # same-named install silently replace a trusted plugin's source while
        # the registry kept the old plugin_id.
        if name in self.registry:
            raise ValueError(
                f"Plugin '{name}' is already installed; uninstall it first."
            )

        # Verify Sandbox compliance before installation
        try:
            PluginSandbox.analyze_ast(source_code)
        except Exception as e:
            raise ValueError(f"Plugin rejected during security scan: {str(e)}")

        # Save code (explicit UTF-8: the loader exec's the file back and
        # platform-default encodings corrupt non-ASCII source).
        plugin_path = os.path.join(self.plugins_dir, f"{name}.py")
        if os.path.dirname(os.path.abspath(plugin_path)) != os.path.abspath(self.plugins_dir):
            raise ValueError("Refusing to install outside the plugins directory.")
        with open(plugin_path, 'w', encoding='utf-8') as f:
            f.write(source_code)
            
        self.registry[name] = {
            "version": version,
            "dependencies": manifest.get("dependencies", []),
            "description": manifest.get("description", ""),
            "author": manifest.get("author", ""),
            "enabled": False,
            "path": plugin_path
        }
        self.save_registry()
        return f"Plugin {name} v{version} installed successfully."

    def enable_plugin(self, name: str):
        if name not in self.registry:
            raise ValueError(f"Plugin {name} not found.")
        self.registry[name]["enabled"] = True
        self.save_registry()
        self._recalculate_active()

    def disable_plugin(self, name: str):
         if name not in self.registry:
             raise ValueError(f"Plugin {name} not found.")
         self.registry[name]["enabled"] = False
         self.save_registry()
         self._recalculate_active()

    def _recalculate_active(self):
        """
        Determines the load order of enabled plugins, resolving dependencies.
        """
        deps = {}
        enabled_plugins = [name for name, info in self.registry.items() if info["enabled"]]
        
        for name in enabled_plugins:
            # Only consider dependencies that are also enabled
            deps[name] = [dep for dep in self.registry[name]["dependencies"] if dep in enabled_plugins]
            
        try:
             self.active_plugins = PluginDependencyResolver.resolve(deps)
        except Exception as e:
             # Revert problematic enables in real-world scenario
             self.active_plugins = []
             raise RuntimeError(f"Failed to resolve dependencies: {str(e)}")

    def load_active_plugins(self) -> Dict[str, Any]:
        """
        Loads all active plugins in the correct dependency order.
        """
        self._recalculate_active()
        loaded_modules = {}
        
        for plugin_name in self.active_plugins:
             path = self.registry[plugin_name]["path"]
             with open(path, 'r') as f:
                 code = f.read()
             
             # Context can include already loaded plugins as dependencies
             context = {"plugins": loaded_modules}
             try:
                 locals_dict = PluginSandbox.execute(code, context)
                 loaded_modules[plugin_name] = locals_dict
             except Exception as e:
                 print(f"Failed to load plugin {plugin_name}: {str(e)}")
                 # In a robust system, might disable the plugin and recalculate
                 
        return loaded_modules
