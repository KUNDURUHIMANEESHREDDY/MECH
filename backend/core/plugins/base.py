"""Base classes for MECH Plugin/Tool System."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("MECH.plugins")


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types, tuples, and sets to JSON-serializable Python natives."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(x) for x in obj]
    return obj


@dataclass
class ParameterSpec:
    """Specification for a tool parameter."""

    type: str
    required: bool = False
    default: Any = None
    description: str = ""
    enum: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type, "required": self.required}
        if self.default is not None:
            result["default"] = self.default
        if self.description:
            result["description"] = self.description
        if self.enum is not None:
            result["enum"] = self.enum
        return result


@dataclass
class ToolManifest:
    """Manifest/metadata for a tool."""

    name: str
    description: str
    parameters: Dict[str, ParameterSpec] = field(default_factory=dict)
    category: str = "interpretability"
    version: str = "1.0.0"
    author: str = "MECH Platform"
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
            "category": self.category,
            "version": self.version,
            "author": self.author,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolManifest":
        params = {
            k: ParameterSpec(**v) if isinstance(v, dict) else v
            for k, v in data.get("parameters", {}).items()
        }
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=params,
            category=data.get("category", "interpretability"),
            version=data.get("version", "1.0.0"),
            author=data.get("author", "MECH Platform"),
            capabilities=data.get("capabilities", []),
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
        )


class BaseTool(ABC):
    """Abstract base class for all MECH tools."""

    def __init__(self, manifest: ToolManifest) -> None:
        self.manifest = manifest
        self._initialized = False

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def description(self) -> str:
        return self.manifest.description

    @property
    def category(self) -> str:
        return self.manifest.category

    @property
    def parameters(self) -> Dict[str, ParameterSpec]:
        return self.manifest.parameters

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass

    def initialize(self) -> None:
        """Initialize the tool (called once on first use)."""
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self) -> None:
        """Override for tool-specific initialization."""
        pass

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and apply defaults to parameters."""
        validated = {}
        for param_name, param_spec in self.parameters.items():
            if param_name in params:
                validated[param_name] = params[param_name]
            elif param_spec.required:
                raise ValueError(f"Required parameter '{param_name}' not provided")
            elif param_spec.default is not None:
                validated[param_name] = param_spec.default
        return validated

    def to_dict(self) -> Dict[str, Any]:
        return self.manifest.to_dict()


class FunctionTool(BaseTool):
    """Tool wrapper for a simple function handler."""

    def __init__(
        self,
        name: str,
        description: str,
        handler: Callable[..., Dict[str, Any]],
        parameters: Dict[str, ParameterSpec],
        category: str = "interpretability",
        version: str = "1.0.0",
    ) -> None:
        manifest = ToolManifest(
            name=name,
            description=description,
            parameters=parameters,
            category=category,
            version=version,
        )
        super().__init__(manifest)
        self._handler = handler

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        validated = self.validate_params(kwargs)
        try:
            raw_result = self._handler(**validated)
            return sanitize_for_json(raw_result)
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", self.name, e)
            return {
                "tool": self.name,
                "status": "error",
                "error": str(e),
            }


class Plugin(ABC):
    """Abstract base class for MECH plugins."""

    def __init__(self, manifest: "PluginManifest") -> None:
        self.manifest = manifest
        self._tools: Dict[str, BaseTool] = {}
        self._initialized = False

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    @abstractmethod
    def register_tools(self, registry: "ToolRegistry") -> None:
        """Register tools with the registry."""
        pass

    def _add_tool(self, tool: BaseTool) -> None:
        """Add a tool to this plugin and register it with the registry."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def initialize(self) -> None:
        """Initialize the plugin."""
        if not self._initialized:
            self._initialize()
            self._initialized = True

    def _initialize(self) -> None:
        """Override for plugin-specific initialization."""
        pass


@dataclass
class PluginManifest:
    """Manifest/metadata for a plugin."""

    name: str
    version: str
    description: str
    author: str = "MECH Platform"
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    entry_point: str = ""
    config_schema: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "capabilities": self.capabilities,
            "dependencies": self.dependencies,
            "entry_point": self.entry_point,
            "config_schema": self.config_schema,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginManifest":
        return cls(**data)