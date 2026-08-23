"""Core Interaction Engine.

Sends prompts to LLMs through the model layer (local GPT-2, Ollama, or OpenAI),
captures responses, and records interaction history.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from .models import (
    InteractionRecord,
    InteractionRequest,
    InteractionResponse,
)


class InteractionEngine:
    """Facade for sending prompts to models and capturing responses."""

    def __init__(self, storage=None) -> None:
        self._storage = storage
        self._history: List[InteractionRecord] = []
        self._llm_engines: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Engine resolution
    # ------------------------------------------------------------------

    def _get_llm_engine(self, backend: str, model: str) -> Any:
        """Resolve an LLM engine for the requested backend."""
        key = f"{backend}:{model}"
        if key in self._llm_engines:
            return self._llm_engines[key]

        engine = None
        if backend == "ollama":
            from backend.agents.llm.ollama_engine import OllamaEngine

            engine = OllamaEngine(model=model)
        elif backend == "openai":
            from backend.agents.llm.openai_engine import OpenAIEngine

            engine = OpenAIEngine(model=model)
        elif backend in ("local", "gpt2"):
            # Local GPT-2 engine from services
            try:
                from backend.services import gpt2_engine

                engine = gpt2_engine
            except (ImportError, AttributeError, OSError) as exc:
                logger.debug("Local GPT-2 engine unavailable: %s", exc)
                engine = None

        self._llm_engines[key] = engine
        return engine

    # ------------------------------------------------------------------
    # Core interaction
    # ------------------------------------------------------------------

    def send_prompt(
        self,
        prompt: str,
        model: str = "gpt2",
        backend: str = "local",
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 256,
        session_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InteractionRecord:
        """Send a prompt to a model and capture the response."""
        request = InteractionRequest(
            prompt=prompt,
            model=model,
            backend=backend,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            metadata=metadata,
        )

        start = time.monotonic()
        response = self._execute(request)
        latency_ms = round((time.monotonic() - start) * 1000, 2)
        response.latency_ms = latency_ms

        record = InteractionRecord(
            request=request,
            response=response,
            session_id=session_id,
        )
        self._history.append(record)

        if self._storage is not None:
            try:
                self._storage.add_interaction(record.to_dict())
            except (OSError, RuntimeError, ValueError) as exc:
                logger.debug("Failed to persist interaction record: %s", exc)

        return record

    def _execute(self, request: InteractionRequest) -> InteractionResponse:
        """Execute a request against the resolved engine."""
        engine = self._get_llm_engine(request.backend, request.model)

        if engine is None:
            return InteractionResponse(
                text="",
                error=f"No engine available for backend '{request.backend}' model '{request.model}'.",
            )

        try:
            if request.backend in ("local", "gpt2"):
                return self._execute_local(engine, request)
            return self._execute_llm(engine, request)
        except Exception as e:
            return InteractionResponse(text="", error=str(e))

    def _execute_local(self, engine: Any, request: InteractionRequest) -> InteractionResponse:
        """Execute against the local GPT-2 engine."""
        if hasattr(engine, "is_available") and not engine.is_available():
            return InteractionResponse(
                text="",
                error="Local GPT-2 engine is not available (torch/transformers not installed).",
            )

        # If generate is True and engine has generate method, use multi-token generation
        if request.generate and hasattr(engine, "generate"):
            result = engine.generate(
                request.prompt,
                request.model,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
            )
            if result.get("status") == "error":
                return InteractionResponse(text="", error=result.get("error", "Local engine error."))

            return InteractionResponse(
                text=result.get("response", ""),
                raw=result,
                usage={
                    "prompt_tokens": len(request.prompt.split()),
                    "completion_tokens": result.get("n_generated", 0),
                    "total_tokens": len(request.prompt.split()) + result.get("n_generated", 0),
                },
            )

        # Fall back to single-token prediction (run_prompt)
        if hasattr(engine, "run_prompt"):
            result = engine.run_prompt(request.prompt)
            if result.get("status") == "error":
                return InteractionResponse(text="", error=result.get("error", "Local engine error."))

            next_token = result.get("next_token", "")
            top5 = result.get("top5", [])
            str_tokens = result.get("str_tokens", [])
            text = "".join(str_tokens) + next_token

            return InteractionResponse(
                text=text,
                raw=result,
                usage={
                    "prompt_tokens": len(str_tokens),
                    "completion_tokens": 1,
                    "total_tokens": len(str_tokens) + 1,
                },
            )

        # Fall back to infer
        if hasattr(engine, "infer"):
            result = engine.infer(request.prompt, request.model)
            tokens = result.get("tokens", [])
            text = result.get("generated_text", "")
            return InteractionResponse(
                text=text,
                raw=result,
                usage={
                    "prompt_tokens": len(tokens),
                    "completion_tokens": 1,
                    "total_tokens": len(tokens) + 1,
                },
            )

        return InteractionResponse(
            text="",
            error="Local engine does not expose run_prompt, infer, or generate.",
        )

    def _execute_llm(self, engine: Any, request: InteractionRequest) -> InteractionResponse:
        """Execute against an LLM engine (Ollama / OpenAI)."""
        # Build a generic prompt that includes system context if provided
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{request.prompt}"

        # The LLM engines expose _call_api; use it directly for generic interaction.
        if hasattr(engine, "_call_api"):
            text = engine._call_api(prompt)
            return InteractionResponse(
                text=text,
                raw={"model": request.model, "backend": request.backend},
                usage={
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(text.split()),
                    "total_tokens": len(prompt.split()) + len(text.split()),
                },
            )

        return InteractionResponse(
            text="",
            error=f"Engine for backend '{request.backend}' does not expose _call_api.",
        )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def list_history(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List interaction history, optionally filtered by session."""
        records = self._history
        if session_id:
            records = [r for r in records if r.session_id == session_id]
        records = records[-limit:]
        return [r.to_dict() for r in records]

    def get_record(self, interaction_id: str) -> Optional[Dict[str, Any]]:
        """Get a single interaction record by ID."""
        for r in self._history:
            if r.interaction_id == interaction_id:
                return r.to_dict()
        return None

    def clear_history(self) -> None:
        """Clear in-memory interaction history."""
        self._history = []

    # ------------------------------------------------------------------
    # Backend / model discovery
    # ------------------------------------------------------------------

    def list_backends(self) -> List[Dict[str, Any]]:
        """List available interaction backends."""
        backends = [
            {
                "id": "local",
                "name": "Local GPT-2",
                "description": "Local GPT-2 inference via torch/transformers",
                "models": ["gpt2", "gpt2-medium", "distilgpt2"],
                "available": True,
            },
            {
                "id": "ollama",
                "name": "Ollama (Local Models)",
                "description": "Local Ollama instance for llama3, mistral, etc.",
                "models": ["llama3", "llama3.1", "mistral", "qwen2.5", "gemma2"],
                "available": True,
            },
            {
                "id": "openai",
                "name": "OpenAI API",
                "description": "OpenAI chat completions (requires OPENAI_API_KEY)",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                "available": True,
            },
        ]
        return backends

    def list_models(self, backend: Optional[str] = None) -> List[Dict[str, Any]]:
        """List models available for a backend (or all backends)."""
        all_models = []
        for b in self.list_backends():
            if backend and b["id"] != backend:
                continue
            for m in b["models"]:
                all_models.append({"backend": b["id"], "model": m, "name": m})
        return all_models