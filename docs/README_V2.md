# MECH Platform v2.0: Mechanistic Interpretability Research Operating System

MECH is a scientific research operating system for discovering, intervening on, and validating neural-network mechanisms. It provides an end-to-end scientific workflow that bridges exploratory interpretability and rigorous causal validation.

---

## 1. Core Research Lifecycle

$$\text{Investigation} \longrightarrow \text{Hypothesis} \longrightarrow \text{Experiment} \longrightarrow \text{Intervention} \longrightarrow \text{Control} \longrightarrow \text{Evidence} \longrightarrow \text{Mechanism} \longrightarrow \text{Claim} \longrightarrow \text{Reproduction}$$

1. **Investigation**: Define a research question, target architecture, and evaluation dataset.
2. **Hypothesis**: Formulate testable causal claims with explicit falsification criteria.
3. **Preregistration**: Cryptographically freeze experimental protocols to prevent post-hoc HARKing.
4. **Intervention**: Execute live weight/activation zero-ablations, mean-ablations, and activation patching.
5. **Negative Control**: Contrast causal shifts against isolated negative control heads (e.g. `L0H0`) to isolate target specificity.
6. **Replication**: Validate effect size stability across multiple independent seeds and prompt distributions.
7. **Evidence & Mechanism**: Synthesize verified findings into inspectable circuit topologies.
8. **Claim Calibration**: Enforce conservative epistemic claim levels (`OBSERVATIONAL`, `CAUSAL_COMPONENT`, `MECHANISTIC_CIRCUIT`).
9. **Reproduction**: Export immutable `.mech` research bundles with SHA-256 manifest verification.

---

## 2. Hardware Support & Validation Matrix

MECH distinguishes three tiers of hardware execution:

| Tier | Category | Validation Status | Scope & Capabilities |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **CPU Execution** | ✅ **PRODUCTION VALIDATED** | GPT-2 small (124M), standard prompt distributions (N=100), whole-model forward-hook ablation scans, prefix attention analysis, deterministic torch seed control. |
| **Tier 2** | **NVIDIA CUDA GPU** | ⚠️ **NOT YET VALIDATED** | GPU execution pathways are architecturally implemented with graceful fallback, but have not been evaluated on dedicated CUDA hardware in this release environment. |
| **Tier 3** | **Large Models / Multi-GPU** | 🛑 **EXPERIMENTAL / UNSUPPORTED** | Distributed multi-GPU execution, models exceeding available host memory (>7B parameters). |

---

## 3. Validated Runtime Environment

- **Python Runtime**: CPython 3.14.6 64-bit (`MSC v.1944 64 bit AMD64`)
- **PyTorch Stack**: PyTorch 2.13.0+cpu (CPU execution mode, deterministic torch seed 42)
- **Hugging Face Stack**: `transformers` 5.12.1, `pydantic` 2.12.5, `fastapi` 0.141.1
- **Desktop Runtime**: Electron 32.3.3, Node.js 20.x, React 18.x, embedded SQLite 3 (`better-sqlite3` 11.x)
- **Supported Workloads**: GPT-2 family (`gpt2`, `gpt2-medium`, `gpt2-large`), Transformer-Lens compatible models, PyTorch Hugging Face architectures.

---

## 4. Key Interfaces

- **Inspectable Mechanism Graph**: Interactive circuit visualization with clickable causal edges detailing evidence IDs, interventions, negative controls, and limitations.
- **Universal Scientific Inspector**: Explains the complete "Why am I seeing this?" derivation chain for every component, edge, and claim with explicit `[COMPUTED]` provenance tags.
- **What Should I Test Next?**: Actionable discriminative experiment recommendations from the critic engine with structured Outcome A / B / C matrices and background queue execution.
- **Side-by-Side Comparison Workspace**: Compare experimental runs (treatments vs controls) and competing circuit mechanisms.
- **1-Click Methodology Drawer**: Instant slide-out panel detailing model versions, prompt construction, intervention techniques, sample sizes, and normalization.
- **Global Quick Search (`Ctrl+K`)**: Rapid indexed search across components, hypotheses, experiments, evidence, mechanisms, and literature benchmarks.

---

## 5. Installation & Distribution

### Windows Portable Application
1. Download `MECH Platform.exe` from `release/win-unpacked/`.
2. Run `MECH Platform.exe` directly (self-contained desktop executable).

### Windows Setup Installer
1. Run `MECH Platform Setup 2.0.0.exe` from `release/`.
2. Follow the NSIS installation wizard to install MECH to your user profile.

---

## 6. Known Scientific Limitations & Epistemic Boundaries

1. **Single-Head Knockout**: Isolated single-head ablations demonstrate component mediation but do not prove complete circuit sufficiency without joint multi-head patching.
2. **Observational vs Causal**: Attention routing patterns and activation correlations are classified as `OBSERVATIONAL_CLAIM` and cannot assert causal mediation without intervention.
3. **Backup Head Compensation**: Redundant circuits (e.g. Backup Name Movers in GPT-2) may buffer output logits when individual components are ablated in isolation.
