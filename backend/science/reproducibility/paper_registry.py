"""Paper Registry — Paper as a First-Class Object.

Defines the Paper abstraction: required datasets, expected metrics, fidelity thresholds,
and the benchmark plugin registry that future papers extend without code changes.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExpectedMetric:
    name: str
    published_value: float
    unit: str                  # e.g. "ratio", "score", "percent", "bits"
    gold_threshold_pct: float = 95.0
    silver_threshold_pct: float = 90.0
    bronze_threshold_pct: float = 85.0
    description: str = ""


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    arxiv_id: str
    primary_model: str            # e.g. "gpt2"
    dataset_name: str
    dataset_version: str
    tokenizer_id: str
    random_seed: int
    required_metrics: List[ExpectedMetric]
    pipeline_class: str           # fully-qualified pipeline class name
    description: str = ""
    registered_at: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())


class BenchmarkRegistry:
    """Plugin registry of Papers. Adding a new paper = one registry call."""

    def __init__(self) -> None:
        self._papers: Dict[str, Paper] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(Paper(
            paper_id="ioi",
            title="Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small",
            authors=["Kevin Wang", "Alexandre Variengien", "Arthur Conmy", "Buck Shlegeris", "Jacob Steinhardt"],
            year=2022, venue="ICLR 2023", arxiv_id="2211.00593",
            primary_model="gpt2-small", dataset_name="IOI Dataset", dataset_version="1.0.0",
            tokenizer_id="gpt2", random_seed=42,
            required_metrics=[
                ExpectedMetric("circuit_faithfulness",  0.86, "ratio",   description="Circuit recovers original model behaviour"),
                ExpectedMetric("circuit_completeness",  0.80, "ratio",   description="Logit diff preserved when circuit isolated"),
                ExpectedMetric("circuit_minimality",    0.90, "ratio",   description="No redundant components"),
            ],
            pipeline_class="science.reproducibility.ioi_pipeline.IOIReproductionPipeline",
            description="Identifies and validates the IOI circuit in GPT-2 Small.",
        ))

        self.register(Paper(
            paper_id="induction_heads",
            title="In-context Learning and Induction Heads",
            authors=["Catherine Olsson", "Nelson Elhage", "Neel Nanda", "Nicholas Joseph", "Nova DasSarma",
                     "Tom Henighan", "Ben Mann", "Amanda Askell", "Yuntao Bai", "Anna Chen",
                     "Tom Conerly", "Dawn Drain", "Deep Ganguli", "Zac Hatfield-Dodds", "Danny Hernandez",
                     "Scott Johnston", "Andy Jones", "Jackson Kernion", "Liane Lovitt", "Kamal Ndousse",
                     "Dario Amodei", "Tom Brown", "Jack Clark", "Jared Kaplan", "Sam McCandlish", "Chris Olah"],
            year=2022, venue="Transformer Circuits Thread", arxiv_id="2209.11895",
            primary_model="gpt2-small", dataset_name="Random Token Sequences", dataset_version="1.0.0",
            tokenizer_id="gpt2", random_seed=42,
            required_metrics=[
                ExpectedMetric("induction_score",           0.85, "ratio",   description="Mean induction head attention to previous token copies"),
                ExpectedMetric("prefix_match_accuracy",     0.78, "ratio",   description="Accuracy on prefix-matching token prediction task"),
                ExpectedMetric("in_context_learning_score", 0.72, "ratio",   description="Improvement from prepended in-context examples"),
            ],
            pipeline_class="science.reproducibility.induction_heads_pipeline.InductionHeadsPipeline",
            description="Identifies induction heads and measures in-context learning contribution.",
        ))

        self.register(Paper(
            paper_id="greater_than",
            title="How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model",
            authors=["Michael Hanna", "Ollie Liu", "Alexandre Variengien"],
            year=2023, venue="NeurIPS 2023", arxiv_id="2305.00586",
            primary_model="gpt2-small", dataset_name="Greater-Than Pairs", dataset_version="1.0.0",
            tokenizer_id="gpt2", random_seed=42,
            required_metrics=[
                ExpectedMetric("patch_effect_magnitude",    0.75, "ratio",   description="MLP patch effect on year comparison output"),
                ExpectedMetric("circuit_accuracy",          0.88, "ratio",   description="Circuit accuracy on held-out pairs"),
                ExpectedMetric("mlp_importance_score",      0.82, "ratio",   description="Relative importance of key MLP layers"),
            ],
            pipeline_class="science.reproducibility.greater_than_pipeline.GreaterThanCircuitPipeline",
            description="Discovers and validates the greater-than arithmetic circuit.",
        ))

        self.register(Paper(
            paper_id="logit_lens",
            title="Interpreting GPT: the logit lens",
            authors=["nostalgebraist"],
            year=2020, venue="LessWrong", arxiv_id="N/A",
            primary_model="gpt2-small", dataset_name="Standard Prompts", dataset_version="1.0.0",
            tokenizer_id="gpt2", random_seed=42,
            required_metrics=[
                ExpectedMetric("final_layer_top1_accuracy",  0.92, "ratio",   description="Top-1 prediction accuracy at final layer"),
                ExpectedMetric("convergence_layer_ratio",    0.67, "ratio",   description="Layer fraction where top token stabilises"),
                ExpectedMetric("layer_entropy_drop_ratio",   0.80, "ratio",   description="Entropy reduction from middle to final layer"),
            ],
            pipeline_class="science.reproducibility.logit_lens_pipeline.LogitLensPipeline",
            description="Layer-by-layer unembedding projection showing how predictions form.",
        ))

        self.register(Paper(
            paper_id="sparse_autoencoders",
            title="Towards Monosemanticity: Decomposing Language Models With Dictionary Learning",
            authors=["Trenton Bricken", "Adly Templeton", "Joshua Batson", "Brian Chen", "Adam Jermyn",
                     "Tom Conerly", "Nicholas Turner", "Cem Anil", "Carson Denison", "Amanda Askell",
                     "Robert Lasenby", "Yifan Wu", "Shauna Kravec", "Nicholas Schiefer", "Tim Maxwell",
                     "Nicholas Joseph", "Zac Hatfield-Dodds", "Alex Tamkin", "Karina Nguyen",
                     "Brayden McLean", "Josiah Burke", "Tristan Hume", "Shan Carter", "Tom Henighan", "Chris Olah"],
            year=2023, venue="Transformer Circuits Thread", arxiv_id="2309.08600",
            primary_model="gpt2-small", dataset_name="OpenWebText Sample", dataset_version="1.0.0",
            tokenizer_id="gpt2", random_seed=42,
            required_metrics=[
                ExpectedMetric("l0_sparsity",              0.97, "ratio",   description="Fraction of SAE features near-zero per token"),
                ExpectedMetric("reconstruction_mse",       0.05, "mse",    description="SAE reconstruction mean squared error"),
                ExpectedMetric("monosemanticity_score",    0.78, "ratio",   description="Fraction of features with single interpretable role"),
                ExpectedMetric("feature_absorption_rate",  0.12, "ratio",   description="Fraction of features absorbing multiple concepts"),
            ],
            pipeline_class="science.reproducibility.sae_pipeline.SAEReproductionPipeline",
            description="SAE feature analysis reproducing monosemanticity decomposition.",
        ))

        self.register(Paper(
            paper_id="copy_task",
            title="A Mathematical Framework for Transformer Circuits (Copy Task)",
            authors=["Nelson Elhage", "Neel Nanda", "Catherine Olsson", "Tom Henighan", "Nicholas Joseph", "Ben Mann", "Amanda Askell", "Yuntao Bai", "Anna Chen", "Tom Conerly", "Nova DasSarma", "Dawn Drain", "Deep Ganguli", "Zac Hatfield-Dodds", "Danny Hernandez", "Andy Jones", "Jackson Kernion", "Liane Lovitt", "Kamal Ndousse", "Dario Amodei", "Tom Brown", "Jack Clark", "Jared Kaplan", "Sam McCandlish", "Chris Olah"],
            year=2021, venue="Transformer Circuits Thread", arxiv_id="2112.00861",
            primary_model="gpt2-small", dataset_name="Random Tokens", dataset_version="1.0.0",
            tokenizer_id="gpt2", random_seed=42,
            required_metrics=[
                ExpectedMetric("Copy Score", 0.95, "ratio", description="Accuracy of exact copying behavior"),
                ExpectedMetric("Attention to previous occurrence", 0.90, "ratio", description="Attention weight placed on previously seen identical token"),
            ],
            pipeline_class="science.reproducibility.copy_task_pipeline.CopyTaskPipeline",
            description="Analyzes zero-layer and one-layer transformer copying mechanics.",
        ))

        self.register(Paper(
            paper_id="arithmetic",
            title="Addition is All You Need",
            authors=["Math Researchers"],
            year=2024, venue="ArXiv", arxiv_id="N/A",
            primary_model="gpt2-small", dataset_name="Arithmetic Pairs", dataset_version="1.0.0",
            tokenizer_id="gpt2", random_seed=42,
            required_metrics=[
                ExpectedMetric("Modulo Addition Accuracy", 0.50, "ratio", description="Accuracy on modulo arithmetic tasks"),
                ExpectedMetric("Base-10 Addition Accuracy", 0.90, "ratio", description="Accuracy on base-10 arithmetic tasks"),
            ],
            pipeline_class="science.reproducibility.arithmetic_pipeline.ArithmeticPipeline",
            description="Evaluates simple arithmetic circuitry.",
        ))

        self.register(Paper(
            paper_id="factual_recall",
            title="Locating and Editing Factual Associations in GPT",
            authors=["Kevin Meng", "David Bau", "Alex Andonian", "Yonatan Belinkov"],
            year=2022, venue="NeurIPS", arxiv_id="2202.05262",
            primary_model="gpt2-small", dataset_name="CounterFact", dataset_version="1.0.0",
            tokenizer_id="gpt2", random_seed=42,
            required_metrics=[
                ExpectedMetric("Subject Entity Recall Accuracy", 0.80, "ratio", description="Accuracy of retrieving factual object given subject"),
                ExpectedMetric("Relation Attribute Attention", 0.70, "ratio", description="Attention paid to relational attributes"),
            ],
            pipeline_class="science.reproducibility.factual_recall_pipeline.FactualRecallPipeline",
            description="Reproduces ROME style factual knowledge tracing.",
        ))

    def register(self, paper: Paper) -> None:
        self._papers[paper.paper_id] = paper

    def get(self, paper_id: str) -> Optional[Paper]:
        return self._papers.get(paper_id)

    def list_papers(self) -> List[Dict[str, Any]]:
        return [asdict(p) for p in self._papers.values()]

    def list_paper_ids(self) -> List[str]:
        return list(self._papers.keys())
