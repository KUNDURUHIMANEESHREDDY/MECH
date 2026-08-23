"""Data models for the LLM Interaction layer."""

from __future__ import annotations

import datetime as _dt
import uuid
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class InteractionRequest:
    """A prompt sent to an LLM through a model engine."""

    def __init__(
        self,
        prompt: str,
        model: str = "gpt2",
        backend: str = "local",
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 256,
        max_new_tokens: int = 64,
        generate: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.backend = backend
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        self.generate = generate
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "model": self.model,
            "backend": self.backend,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_new_tokens": self.max_new_tokens,
            "generate": self.generate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionRequest":
        return cls(
            prompt=data.get("prompt", ""),
            model=data.get("model", "gpt2"),
            backend=data.get("backend", "local"),
            system_prompt=data.get("system_prompt"),
            temperature=data.get("temperature", 0.2),
            max_tokens=data.get("max_tokens", 256),
            max_new_tokens=data.get("max_new_tokens", 64),
            generate=data.get("generate", True),
            metadata=data.get("metadata"),
        )


class InteractionResponse:
    """The captured response from an LLM."""

    def __init__(
        self,
        text: str,
        raw: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        self.text = text
        self.raw = raw or {}
        self.usage = usage or {}
        self.latency_ms = latency_ms
        self.error = error

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "raw": self.raw,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionResponse":
        return cls(
            text=data.get("text", ""),
            raw=data.get("raw"),
            usage=data.get("usage"),
            latency_ms=data.get("latency_ms"),
            error=data.get("error"),
        )


class InteractionRecord:
    """A persisted interaction between a user and a model."""

    def __init__(
        self,
        request: InteractionRequest,
        response: InteractionResponse,
        session_id: str = "default",
        interaction_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> None:
        self.interaction_id = interaction_id or _new_id("interaction")
        self.session_id = session_id
        self.request = request
        self.response = response
        self.created_at = created_at or _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "session_id": self.session_id,
            "request": self.request.to_dict(),
            "response": self.response.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InteractionRecord":
        return cls(
            interaction_id=data.get("interaction_id"),
            session_id=data.get("session_id", "default"),
            request=InteractionRequest.from_dict(data.get("request", {})),
            response=InteractionResponse.from_dict(data.get("response", {})),
            created_at=data.get("created_at"),
        )


class ExperimentSpec:
    """A controlled experiment: run the same inputs across models/backends."""

    def __init__(
        self,
        name: str,
        prompts: List[str],
        models: Optional[List[str]] = None,
        backends: Optional[List[str]] = None,
        control_prompts: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
        metrics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.prompts = prompts
        self.models = models or ["gpt2"]
        self.backends = backends or ["local"]
        self.control_prompts = control_prompts or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.metrics = metrics or ["response_length", "top_token", "latency_ms"]
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "prompts": self.prompts,
            "models": self.models,
            "backends": self.backends,
            "control_prompts": self.control_prompts,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentSpec":
        return cls(
            name=data.get("name", "untitled experiment"),
            prompts=data.get("prompts", []),
            models=data.get("models"),
            backends=data.get("backends"),
            control_prompts=data.get("control_prompts"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 64),
            metrics=data.get("metrics"),
            metadata=data.get("metadata"),
        )


class ExperimentResult:
    """Results from running a controlled experiment."""

    def __init__(
        self,
        experiment_id: str,
        spec: ExperimentSpec,
        runs: List[Dict[str, Any]],
        summary: Dict[str, Any],
        created_at: Optional[str] = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.spec = spec
        self.runs = runs
        self.summary = summary
        self.created_at = created_at or _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "spec": self.spec.to_dict(),
            "runs": self.runs,
            "summary": self.summary,
            "created_at": self.created_at,
        }


class ComparisonSpec:
    """Compare behavior between different inputs or models."""

    def __init__(
        self,
        prompt: str,
        models: List[str],
        backends: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prompt = prompt
        self.models = models
        self.backends = backends or ["local"] * len(models)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "models": self.models,
            "backends": self.backends,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparisonSpec":
        return cls(
            prompt=data.get("prompt", ""),
            models=data.get("models", []),
            backends=data.get("backends"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 64),
            metadata=data.get("metadata"),
        )


class ComparisonResult:
    """Results from comparing behavior across models."""

    def __init__(
        self,
        comparison_id: str,
        spec: ComparisonSpec,
        outputs: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        created_at: Optional[str] = None,
    ) -> None:
        self.comparison_id = comparison_id
        self.spec = spec
        self.outputs = outputs
        self.metrics = metrics
        self.created_at = created_at or _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "spec": self.spec.to_dict(),
            "outputs": self.outputs,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }


class InspectionSpec:
    """Inspect model behavior across a set of probe inputs."""

    def __init__(
        self,
        prompt: str,
        model: str = "gpt2",
        backend: str = "local",
        probe_inputs: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.backend = backend
        self.probe_inputs = probe_inputs or []
        self.metrics = metrics or [
            "token_probability",
            "top_tokens",
            "attention_strength",
            "response_length",
        ]
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "model": self.model,
            "backend": self.backend,
            "probe_inputs": self.probe_inputs,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InspectionSpec":
        return cls(
            prompt=data.get("prompt", ""),
            model=data.get("model", "gpt2"),
            backend=data.get("backend", "local"),
            probe_inputs=data.get("probe_inputs"),
            metrics=data.get("metrics"),
            metadata=data.get("metadata"),
        )


class InspectionResult:
    """Results from inspecting model behavior across probe inputs."""

    def __init__(
        self,
        inspection_id: str,
        spec: InspectionSpec,
        probes: List[Dict[str, Any]],
        insights: List[str],
        created_at: Optional[str] = None,
    ) -> None:
        self.inspection_id = inspection_id
        self.spec = spec
        self.probes = probes
        self.insights = insights
        self.created_at = created_at or _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "spec": self.spec.to_dict(),
            "probes": self.probes,
            "insights": self.insights,
            "created_at": self.created_at,
        }