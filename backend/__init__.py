# MECH Backend Package - Consolidated Research Platform

# API
from . import api

# Core Systems
from . import core

# Interpretability & Discovery
from . import interpretability
from . import discovery

# Benchmarking & Reproductions
from . import benchmarking
from . import reproductions

# Science & Statistics
from . import science

# Validation & Rigor
from . import validation

# Runtime & Execution
from . import runtime

# Agents & AI
from . import agents

# Knowledge Graph
from . import knowledge_graph

# Platform & SDK
from . import platform
from . import sdk

# Services
from . import services

# Research & Datasets
from . import research
from . import research_platform
from . import datasets

# UI & Visualization
from . import ui

# Storage & Plugins
from . import storage
from . import plugins

# Analysis & Experiments
from . import analysis
from . import experiments

__all__ = [
    "api", "core", "interpretability", "discovery",
    "benchmarking", "reproductions", "science",
    "validation", "runtime", "agents", "knowledge_graph",
    "platform", "sdk", "services", "research",
    "research_platform", "datasets", "ui", "storage",
    "plugins", "analysis", "experiments"
]
