# Developer Guide

Welcome to the development guide for the Autonomous Research Platform.

## Architecture Overview
Please read `architecture.md` first.

## Setting up the Environment
1. Clone the repository.
2. `pip install -r requirements.txt` (Backend).
3. `npm install` (Frontend).

## Adding a New Paper Benchmark
To add a new benchmark to the reproducibility suite:
1. Create a new pipeline in `backend/science/reproducibility/` (e.g. `my_paper_pipeline.py`).
2. Inherit from the base pipeline or implement a standard `run(model_id)` method.
3. Register the paper and its expected metrics in `backend/science/reproducibility/paper_registry.py`.

## Running Tests
Tests are layered:
- `pytest tests/unit/`
- `pytest tests/integration/`
- `pytest tests/gpu/`
- `pytest tests/reproduction/` (Runs the full `BenchmarkRunner`)
