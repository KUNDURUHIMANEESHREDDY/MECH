"""Plugin Service.

Discovers, validates plugin manifests, and registers dynamic plugins.
"""

from __future__ import annotations

from typing import Any, Dict, List


class PluginService:
    """Service for discovering and validating interpretability plugins."""

    def validate_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        required = ["id", "name", "version", "minimumRuntimeVersion"]
        for k in required:
            if k not in manifest:
                return {"valid": False, "error": f"Missing manifest key: {k}"}
        return {"valid": True}

    def register_plugin(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        check = self.validate_manifest(manifest)
        if not check["valid"]:
            return check
        return {
            "status": "registered",
            "plugin_id": manifest["id"],
            "name": manifest["name"],
            "version": manifest["version"],
        }
