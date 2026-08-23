"""Plugin Manifest Schema for MECH Platform."""

from __future__ import annotations

import json
import jsonschema
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.core.plugins.base import ParameterSpec, PluginManifest, ToolManifest


# JSON Schema for Tool Manifest
TOOL_MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["name", "description"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "description": {"type": "string", "minLength": 1},
        "category": {"type": "string", "default": "interpretability"},
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$", "default": "1.0.0"},
        "author": {"type": "string", "default": "MECH Platform"},
        "capabilities": {"type": "array", "items": {"type": "string"}, "default": []},
        "dependencies": {"type": "array", "items": {"type": "string"}, "default": []},
        "tags": {"type": "array", "items": {"type": "string"}, "default": []},
        "parameters": {
            "type": "object",
            "patternProperties": {
                ".*": {
                    "type": "object",
                    "required": ["type"],
                    "properties": {
                        "type": {"type": "string", "enum": ["string", "integer", "number", "boolean", "object", "array"]},
                        "required": {"type": "boolean", "default": False},
                        "default": {},
                        "description": {"type": "string", "default": ""},
                        "enum": {"type": "array", "items": {}},
                    },
                }
            },
            "default": {},
        },
    },
}


# JSON Schema for Plugin Manifest
PLUGIN_MANIFEST_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["name", "version", "description"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
        "description": {"type": "string", "minLength": 1},
        "author": {"type": "string", "default": "MECH Platform"},
        "capabilities": {"type": "array", "items": {"type": "string"}, "default": []},
        "dependencies": {"type": "array", "items": {"type": "string"}, "default": []},
        "entry_point": {"type": "string", "default": ""},
        "config_schema": {"type": "object", "default": {}},
        "tags": {"type": "array", "items": {"type": "string"}, "default": []},
    },
}


class ManifestValidator:
    """Validates plugin and tool manifests."""

    def __init__(self) -> None:
        self._tool_validator = jsonschema.Draft7Validator(TOOL_MANIFEST_SCHEMA)
        self._plugin_validator = jsonschema.Draft7Validator(PLUGIN_MANIFEST_SCHEMA)

    def validate_tool(self, manifest: Union[ToolManifest, Dict[str, Any]]) -> List[str]:
        """Validate a tool manifest. Returns list of errors (empty if valid)."""
        data = asdict(manifest) if is_dataclass(manifest) else manifest
        errors = []
        for error in self._tool_validator.iter_errors(data):
            errors.append(f"{'.'.join(str(p) for p in error.path)}: {error.message}")
        return errors

    def validate_plugin(self, manifest: Union[PluginManifest, Dict[str, Any]]) -> List[str]:
        """Validate a plugin manifest. Returns list of errors (empty if valid)."""
        data = asdict(manifest) if is_dataclass(manifest) else manifest
        errors = []
        for error in self._plugin_validator.iter_errors(data):
            errors.append(f"{'.'.join(str(p) for p in error.path)}: {error.message}")
        return errors


def load_manifest_from_file(path: Path) -> Dict[str, Any]:
    """Load a manifest from a JSON or YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    content = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml

            return yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML required for YAML manifest files")
    elif path.suffix == ".json":
        return json.loads(content)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}")


def save_manifest_to_file(manifest: Union[ToolManifest, PluginManifest], path: Path) -> None:
    """Save a manifest to a JSON file."""
    data = asdict(manifest)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def create_tool_manifest(
    name: str,
    description: str,
    parameters: Dict[str, Dict[str, Any]],
    category: str = "interpretability",
    version: str = "1.0.0",
    **kwargs: Any,
) -> ToolManifest:
    """Create a ToolManifest from simplified parameter dict."""
    param_specs = {}
    for param_name, param_def in parameters.items():
        param_specs[param_name] = ParameterSpec(
            type=param_def.get("type", "string"),
            required=param_def.get("required", False),
            default=param_def.get("default"),
            description=param_def.get("description", ""),
            enum=param_def.get("enum"),
        )
    return ToolManifest(
        name=name,
        description=description,
        parameters=param_specs,
        category=category,
        version=version,
        **kwargs,
    )


def create_plugin_manifest(
    name: str,
    version: str,
    description: str,
    **kwargs: Any,
) -> PluginManifest:
    """Create a PluginManifest."""
    return PluginManifest(
        name=name,
        version=version,
        description=description,
        **kwargs,
    )