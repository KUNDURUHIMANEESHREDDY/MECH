"""Neuron Inspector - A neural network activation analysis toolkit.

Architecture:

    Runtime

    |

    ActivationRepository

    |

    Inspectors
        | - NeuronInspector
        | - AttentionInspector
        | - ResidualInspector
        | - LayerInspector
        | - TokenInspector
        | - LogitInspector

    |

    REST API (v1)

    |

    Frontend
"""

# Eager imports here would pull torch/transformers (~11s) into every app/API
# startup (logit_lens -> runtime.logits and friends). Names are resolved on
# first attribute access instead; direct submodule imports are unaffected.

_LAZY_EXPORTS = {
    "Statistics": (".models", "Statistics"),
    "NeuronInspection": (".models", "NeuronInspection"),
    "AttentionInspection": (".models", "AttentionInspection"),
    "ResidualInspection": (".models", "ResidualInspection"),
    "LayerInspection": (".models", "LayerInspection"),
    "TokenInspection": (".models", "TokenInspection"),
    "LogitInspection": (".models", "LogitInspection"),
    "HeadRanking": (".models", "HeadRanking"),
    "ActivationSearchResult": (".models", "ActivationSearchResult"),
    "VisualizationDTO": (".models", "VisualizationDTO"),
    "NeuronData": (".models", "NeuronData"),
    "AttentionData": (".models", "AttentionData"),
    "ResidualData": (".models", "ResidualData"),
    "LayerData": (".models", "LayerData"),
    "HeatmapData": (".models", "HeatmapData"),
    "StatisticsComputer": (".statistics", "StatisticsComputer"),
    "Runtime": (".runtime", "Runtime"),
    "MockRuntime": (".mock_runtime", "MockRuntime"),
    "get_default_runtime": (".mock_runtime", "get_default_runtime"),
    "ActivationRepository": (".repository", "ActivationRepository"),
    "NeuronInspector": (".neuron_inspector", "NeuronInspector"),
    "AttentionInspector": (".attention_inspector", "AttentionInspector"),
    "ResidualInspector": (".residual_inspector", "ResidualInspector"),
    "LayerInspector": (".layer_inspector", "LayerInspector"),
    "TokenInspector": (".token_inspector", "TokenInspector"),
    "LogitInspector": (".logit_inspector", "LogitInspector"),
    "VisualizationGenerator": (".heatmap", "VisualizationGenerator"),
    "HeatmapGenerator": (".heatmap", "HeatmapGenerator"),
    "app": (".api", "app"),
    "algorithms": (".algorithms", None),
    "sae": (".sae", None),
    "causal": (".causal", None),
    "semantics": (".semantics", None),
}

__version__ = "2.0.0"
__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    entry = _LAZY_EXPORTS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(entry[0], __name__)
    value = module if entry[1] is None else getattr(module, entry[1])
    globals()[name] = value
    return value
