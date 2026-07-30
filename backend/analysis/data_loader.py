"""
Data loader for activation data.

Loads activation data from various formats (JSON, NumPy, HDF5) and
converts them into the MockModelData format used by the inspectors.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

from backend.interpretability.data_generator import MockModelData


class ActivationDataLoader:
    """Loads activation data from files and converts to MockModelData.

    Supports loading from:
    - JSON files with activation matrices
    - NumPy .npz files
    - Pickle files

    Parameters
    ----------
    data_path : str, optional
        Path to the data file or directory.
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path

    def load_json(self, path: str) -> MockModelData:
        """Load activation data from a JSON file.

        Expected JSON format:
        {
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "seq_len": 16,
            "layers": [
                {
                    "name": "layer_0",
                    "activations": [[...], ...],
                    "attention": [[[...], ...], ...],
                    "residual": [[...], ...]
                },
                ...
            ]
        }
        """
        with open(path, "r") as f:
            data = json.load(f)

        model = MockModelData(
            num_layers=data.get("num_layers", len(data["layers"])),
            num_heads=data.get("num_heads", 12),
            hidden_dim=data.get("hidden_dim", 768),
            seq_len=data.get("seq_len", 16),
        )

        # Override with loaded data
        model._activations = {}
        model._attention = {}
        model._residuals = {}

        for layer_data in data["layers"]:
            name = layer_data["name"]
            model._activations[name] = np.array(layer_data["activations"])
            model._attention[name] = np.array(layer_data["attention"])
            model._residuals[name] = np.array(layer_data["residual"])

        model.layer_names = [ld["name"] for ld in data["layers"]]
        model.num_layers = len(model.layer_names)
        model.hidden_dim = model._activations[model.layer_names[0]].shape[-1]
        model.seq_len = model._activations[model.layer_names[0]].shape[0]

        return model

    def load_npz(self, path: str) -> MockModelData:
        """Load activation data from a NumPy .npz file.

        Expected keys: activations, attention, residuals, layer_names
        """
        data = np.load(path, allow_pickle=True)

        activations = data["activations"]  # (num_layers, seq_len, hidden_dim)
        attention = data["attention"]  # (num_layers, num_heads, seq_len, seq_len)
        residuals = data["residuals"]  # (num_layers, seq_len, hidden_dim)
        layer_names = data["layer_names"].tolist()

        num_layers, seq_len, hidden_dim = activations.shape
        num_heads = attention.shape[1]

        model = MockModelData(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            seq_len=seq_len,
        )

        model._activations = {}
        model._attention = {}
        model._residuals = {}

        for i, name in enumerate(layer_names):
            model._activations[name] = activations[i]
            model._attention[name] = attention[i]
            model._residuals[name] = residuals[i]

        model.layer_names = layer_names
        return model

    def save_json(self, model: MockModelData, path: str) -> None:
        """Save model data to a JSON file."""
        data = {
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "hidden_dim": model.hidden_dim,
            "seq_len": model.seq_len,
            "layers": [],
        }

        for name in model.layer_names:
            layer_data = {
                "name": name,
                "activations": model._activations[name].tolist(),
                "attention": model._attention[name].tolist(),
                "residual": model._residuals[name].tolist(),
            }
            data["layers"].append(layer_data)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_npz(self, model: MockModelData, path: str) -> None:
        """Save model data to a NumPy .npz file."""
        activations = np.stack(
            [model._activations[name] for name in model.layer_names]
        )
        attention = np.stack(
            [model._attention[name] for name in model.layer_names]
        )
        residuals = np.stack(
            [model._residuals[name] for name in model.layer_names]
        )
        layer_names = np.array(model.layer_names)

        np.savez(
            path,
            activations=activations,
            attention=attention,
            residuals=residuals,
            layer_names=layer_names,
        )

    def load(self, path: Optional[str] = None) -> MockModelData:
        """Auto-detect file format and load."""
        path = path or self.data_path
        if path is None:
            raise ValueError("No data path provided")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".json":
            return self.load_json(path)
        elif ext in (".npz", ".npy"):
            return self.load_npz(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
