from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from backend.neuron_inspector import GPT2Model

# Path bootstrap.
#
# The two imports below resolve against different roots:
#   "backend.neuron_inspector" -> <repo>/backend/neuron_inspector.py  => repo root
#   "storage.database"         -> <repo>/backend/storage/database.py => backend/
# The original code added only frontend/, which is neither, so the process died
# at import with ModuleNotFoundError: No module named 'storage'.
#
# Roots come from the resolved __file__, not the working directory: Electron
# spawns this with an app-root cwd that differs from the script location.
# resolve() also means a symlinked launch still lands on the real tree.
#
# Order matters. sys.path is searched left to right, so the repo root must come
# before backend/ (or a stray frontend/ entry would shadow the backend package)
# and frontend/ goes last because nothing imports from it.
def _candidate_roots() -> list[Path]:
    script = Path(__file__).resolve()
    frontend = script.parents[1]        # .../<repo>/frontend
    repo = frontend.parent              # .../<repo>

    roots: list[Path] = [repo, repo / "backend"]

    # Packaged layout: electron-builder copies ../backend -> resources/backend
    # and the app root is process.resourcesPath, so the tree is a sibling of
    # the app root rather than a parent. MECH_APP_ROOT names that app root.
    override = os.environ.get("MECH_APP_ROOT", "").strip()
    if override:
        app_root = Path(override).expanduser().resolve()
        roots += [app_root.parent, app_root.parent / "backend",
                  app_root, app_root / "backend"]

    roots.append(frontend)

    # Dedupe while preserving precedence: the first occurrence of a path wins.
    seen: set[str] = set()
    ordered: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            ordered.append(root)
    return ordered


def _bootstrap_sys_path() -> list[Path]:
    """Put the import roots on sys.path, highest precedence first.

    Roots already present are *moved*, not skipped. A pre-existing entry at the
    wrong precedence — from PYTHONPATH, a parent process, or a test harness —
    would otherwise shadow the backend package, which is the exact failure this
    bootstrap exists to prevent.
    """
    added: list[Path] = []
    for root in reversed(_candidate_roots()):   # reversed => insert(0) keeps order
        if not root.is_dir():
            continue
        entry = str(root)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
        added.append(root)
    return added


_bootstrap_sys_path()

from storage.database import DesktopStorage, StorageError  # noqa: E402


class NeuronInspectorController:
    """Dispatches IPC requests — all real model inference via GPT2Model."""

    def __init__(self):
        self._gpt2: "GPT2Model | None" = None

    def _get_gpt2(self) -> "GPT2Model":
        """Import and construct the model on first neuron request.

        The import is deliberately deferred. backend.neuron_inspector pulls in
        transformer_lens -> datasets, and a broken or missing ML dependency
        should only break neuron inspection — not settings, projects, or
        workspace.describe, none of which need a model. Importing it at module
        scope made the whole sidecar unbootable.
        """
        if self._gpt2 is None:
            try:
                from backend.neuron_inspector import GPT2Model as _GPT2Model
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"GPT-2 is unavailable for neuron inspection: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            self._gpt2 = _GPT2Model("gpt2-small")
        return self._gpt2

    def handle(self, method: str, params: dict[str, Any]) -> Any:
        # Model info (from GPT2Model, not MockRuntime)
        if method == "neuron.modelInfo":
            gpt2 = self._get_gpt2()
            return {
                "num_layers": gpt2.num_layers,
                "num_heads": gpt2.num_heads,
                "hidden_dim": gpt2.hidden_dim,
                "vocab_size": gpt2.vocab_size,
                "layer_names": [f"blocks.{i}" for i in range(gpt2.num_layers)],
            }.model_dump()

        # Prompt inspection (uses real GPT-2 Small via TransformerLens)
        if method == "prompt.run":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", ""))
            top_k = int(params.get("top_k", 20))
            model = self._gpt2.run(prompt_text)
            tokens = model.str_tokens
            predictions = model.top_k_predictions(top_k)
            top1 = predictions[0] if predictions else None
            return {
                "prompt": prompt_text,
                "tokens": tokens,
                "top1": top1,
                "predictions": predictions,
            }

        if method == "prompt.tokenLookup":
            self._get_gpt2()
            token_str = str(params.get("token", ""))
            if not token_str.strip():
                return {"error": "No token provided"}
            result = self._gpt2.token_logit(token_str)
            if result.get("logit") is None and self._gpt2.last_position_logits is None:
                result["note"] = "Run a prompt first to get logits in context"
            return result

        if method == "prompt.ioi":
            self._get_gpt2()
            return self._gpt2.ioi_analysis()

        if method == "prompt.cacheShapes":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "The capital of France is"))
            self._gpt2.run(prompt_text)
            return {
                "prompt": prompt_text,
                "tokens": self._gpt2.str_tokens,
                "layers": self._gpt2.cache_shapes(),
            }

        if method == "prompt.ablate":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "The capital of France is"))
            layer = int(params.get("layer", 0))
            head = int(params.get("head", 0))
            return self._gpt2.run_with_head_ablation(prompt_text, layer, head)

        if method == "prompt.multiAblate":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "When Mary and John went to the store, John gave the bag to"))
            heads_raw = params.get("heads", [[9, 9], [9, 6], [10, 0]])
            heads = [(int(h[0]), int(h[1])) for h in heads_raw]
            return self._gpt2.run_with_multi_ablation(prompt_text, heads)

        if method == "prompt.attentionPattern":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "The capital of France is"))
            layer = int(params.get("layer", 0))
            head = int(params.get("head", 0))
            self._gpt2.run(prompt_text)
            pattern = self._gpt2.attention_pattern(layer, head)
            pattern["prompt"] = prompt_text
            pattern["tokens"] = self._gpt2.str_tokens
            return pattern

        if method == "prompt.patchingMatrix":
            self._get_gpt2()
            prompt_text = str(
                params.get("prompt", "When Mary and John went to the store, John gave the bag to")
            )
            return self._gpt2.compute_patching_matrix(prompt_text)

        if method == "prompt.logitLens":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "The capital of France is"))
            top_k = int(params.get("top_k", 5))
            target_token = params.get("target_token", None)
            target_token = str(target_token) if target_token else None
            return self._gpt2.logit_lens(prompt_text, top_k, target_token)

        if method == "prompt.neuronInspect":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "The capital of France is"))
            layer = int(params.get("layer", 0))
            neuron = int(params.get("neuron", 0))
            target_token = params.get("target_token", None)
            target_token = str(target_token) if target_token else None
            return self._gpt2.neuron_inspect(prompt_text, layer, neuron, target_token)

        if method == "prompt.neuronSearch":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "The capital of France is"))
            layer = int(params.get("layer", 0))
            top_k = int(params.get("top_k", 20))
            return self._gpt2.neuron_search(prompt_text, layer, top_k)

        if method == "prompt.correlatedNeurons":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "The capital of France is"))
            layer = int(params.get("layer", 0))
            neuron = int(params.get("neuron", 0))
            top_k = int(params.get("top_k", 20))
            return self._gpt2.correlated_neurons(prompt_text, layer, neuron, top_k)

        if method == "prompt.neuronEvolution":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "The capital of France is"))
            layer = int(params.get("layer", 0))
            neuron = int(params.get("neuron", 0))
            token_index = int(params.get("token_index", -1))
            return self._gpt2.neuron_evolution(prompt_text, layer, neuron, token_index)

        if method == "prompt.datasetActivation":
            self._get_gpt2()
            prompts_raw = params.get("prompts", None)
            prompts_list: list[str] | None = (
                list(prompts_raw) if isinstance(prompts_raw, list) else None
            )
            layer = int(params.get("layer", 0))
            neuron = int(params.get("neuron", 0))
            top_k = int(params.get("top_k", 50))
            return self._gpt2.dataset_activation(prompts_list, layer, neuron, top_k)

        if method == "prompt.predictionTrace":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "The capital of France is"))
            target_token = params.get("target_token", None)
            target_token = str(target_token) if target_token else None
            layer_for_heads = params.get("layer_for_heads", None)
            layer_for_heads = int(layer_for_heads) if layer_for_heads is not None else None
            return self._gpt2.prediction_trace(prompt_text, target_token, layer_for_heads)

        if method == "prompt.circuitTrace":
            self._get_gpt2()
            prompt_text = str(params.get("prompt", "The capital of France is"))
            target_token = params.get("target_token", None)
            target_token = str(target_token) if target_token else None
            return self._gpt2.circuit_trace(prompt_text, target_token)

        if method == "prompt.promptCompare":
            self._get_gpt2()
            prompt_a = str(params.get("prompt_a", "The capital of France is"))
            prompt_b = str(params.get("prompt_b", "The capital of Germany is"))
            top_k = int(params.get("top_k", 10))
            return self._gpt2.prompt_compare(prompt_a, prompt_b, top_k)

        if method == "prompt.runExperiment":
            self._get_gpt2()
            prompts = params.get("prompts", [])
            if not isinstance(prompts, list) or len(prompts) == 0:
                return {"error": "At least one prompt required."}
            config = params.get("config", {})
            return self._gpt2.run_experiment(prompts, config)

        raise StorageError(f"Unknown method: {method}")


_neuron_controller = NeuronInspectorController()


def handle_request(storage: DesktopStorage, method: str, params: dict[str, Any]) -> Any:
    if method == "ping":
        return {"ok": True, "storage": str(storage.db_path)}
    if method == "settings.get":
        return storage.get_settings()
    if method == "settings.update":
        return storage.update_settings(params.get("settings", {}))
    if method == "projects.list":
        return storage.list_recent_projects(limit=int(params.get("limit", 10)))
    if method == "projects.addRecent":
        return storage.add_recent_project(
            path=str(params.get("path", "")),
            name=params.get("name"),
        )
    if method == "recentFiles.list":
        return storage.list_recent_files(limit=int(params.get("limit", 20)))
    if method == "recentFiles.add":
        return storage.add_recent_file(
            path=str(params.get("path", "")),
            project_path=params.get("projectPath"),
        )
    if method == "workspace.describe":
        return storage.describe_workspace(path=str(params.get("path", "")))

    # Route neuron inspector methods
    if method.startswith(("neuron.", "attention.", "residual.", "layer.", "token.", "logit.", "viz.", "prompt.")):
        return _neuron_controller.handle(method, params)

    raise StorageError(f"Unknown method: {method}")


def write_response(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def run(db_path: Path) -> None:
    storage = DesktopStorage(db_path)
    storage.initialize()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            result = handle_request(
                storage=storage,
                method=str(request.get("method", "")),
                params=request.get("params") or {},
            )
            write_response({"id": request_id, "result": result})
        except Exception as error:  # IPC boundary: serialize all failures.
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            write_response(
                {
                    "id": request_id,
                    "error": {
                        "code": error.__class__.__name__,
                        "message": str(error),
                    },
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural Debugger desktop service")
    parser.add_argument("--db", required=True, type=Path)
    args = parser.parse_args()
    run(args.db)


if __name__ == "__main__":
    main()
