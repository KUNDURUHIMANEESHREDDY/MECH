# Neuron Inspector

A neural network activation analysis toolkit for inspecting neurons, attention heads, and residual streams in transformer-based models.

## Overview

The Neuron Inspector provides a comprehensive framework for analyzing model internals:

- **Neuron Inspector** — Inspect individual neurons: neuron ID, activation value, layer, statistics, and top activating tokens
- **Attention Inspector** — Inspect attention heads: head index, weight matrix, importance score, shape, and statistics
- **Residual Stream Inspector** — Inspect residual vectors: residual vector, layer, statistics, L2 norm, and relative contribution
- **Activation Viewer API** — JSON REST API with `GetNeuron()`, `GetAttention()`, `GetResidual()`, and `GetLayer()` endpoints
- **Heatmap Data** — Prepare JSON heatmap data for frontend visualization of neurons, attention, and residuals
- **Statistics Compute** — Compute max, mean, variance, sparsity, and additional statistics

## Project Structure

```
neuron-inspector/
├── backend/
│   └── interpretability/
│       ├── __init__.py          # Package exports
│       ├── models.py            # Pydantic data models
│       ├── statistics.py        # StatisticsComputer (max, mean, variance, sparsity)
│       ├── data_generator.py    # Mock model data generator
│       ├── neuron_inspector.py  # Neuron Inspector
│       ├── attention_inspector.py  # Attention Inspector
│       ├── residual_inspector.py   # Residual Stream Inspector
│       ├── heatmap.py           # Heatmap data generator
│       └── api.py               # FastAPI Activation Viewer API
├── analysis/
│   ├── __init__.py
│   ├── data_loader.py           # Load/save activation data (JSON, NPZ)
│   └── experiment_runner.py     # Experiment runner and result collection
├── experiments/
│   ├── __init__.py
│   ├── config.py                # Experiment configuration
│   ├── neuron_experiment.py     # Pre-built neuron experiments
│   └── attention_experiment.py  # Pre-built attention experiments
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared test fixtures
│   ├── test_statistics.py       # Statistics tests
│   ├── test_neuron_inspector.py # Neuron Inspector tests
│   ├── test_attention_inspector.py  # Attention Inspector tests
│   ├── test_residual_inspector.py   # Residual Inspector tests
│   ├── test_heatmap.py          # Heatmap tests
│   └── test_api.py              # API endpoint tests
├── requirements.txt
├── run_api.py                   # API server entry point
├── main.py                      # Analysis runner entry point
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the API Server

```bash
python run_api.py
# or with custom port
python run_api.py --port 9000 --reload
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

### Run Analysis

```bash
python main.py
# or with custom parameters
python main.py --layer 5 --top-k 20 --token 3
```

### Run Tests

```bash
pytest tests/ -v
```

## API Endpoints

### GetNeuron()
```
GET /api/neuron?layer_index=0&neuron_index=5&token_index=0
GET /api/neuron/batch?layer_index=0&neuron_indices=0,1,2
GET /api/neuron/top?layer_index=0&top_k=10
```

Returns: `NeuronData` — neuron_id, activation, layer, statistics, activation_history, top_tokens

### GetAttention()
```
GET /api/attention?layer_index=0&head_index=0&token_index=3
GET /api/attention/all?layer_index=0
GET /api/attention/importance?layer_index=0
```

Returns: `AttentionData` — head, matrix, importance, shape, statistics, top_connections

### GetResidual()
```
GET /api/residual?layer_index=0&token_index=0
GET /api/residual/all?token_index=0
GET /api/residual/norms?token_index=0
```

Returns: `ResidualData` — residual_vector, layer, statistics, norm, contribution

### GetLayer()
```
GET /api/layer?layer_index=0&include_neurons=true&include_attention=true
```

Returns: `LayerData` — layer, layer_type, neurons, attention_heads, residual, statistics

### Heatmap Data
```
GET /api/heatmap/neuron?layer_index=0&top_k=20
GET /api/heatmap/attention?layer_index=0&head_index=0&token_index=3
GET /api/heatmap/residual?token_index=0&num_dims=64
GET /api/heatmaps?layer_index=0&head_index=0&token_index=0
```

Returns: `HeatmapData` — heatmap_type, title, data, x_labels, y_labels, color_scale, metadata

## Statistics

The `StatisticsComputer` computes:
- **max** — Maximum value
- **mean** — Arithmetic mean
- **variance** — Population variance
- **sparsity** — Fraction of near-zero elements (default threshold: 1e-4)
- **min** — Minimum value
- **std** — Standard deviation
- **median** — Median value
- **l1_norm** — L1 (Manhattan) norm
- **l2_norm** — L2 (Euclidean) norm
- **num_elements** — Total number of elements

## Definition of Done

- [x] Inspect neurons (Neuron ID, Activation, Layer, Statistics)
- [x] Inspect attention (Head, Matrix, Importance, Shape)
- [x] Inspect residuals (Residual vector)
- [x] JSON APIs ready (GetNeuron, GetAttention, GetResidual, GetLayer)
- [x] Heatmap data preparation (JSON for frontend)
- [x] Statistics computation (max, mean, variance, sparsity)
- [x] Tests for all components
