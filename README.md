<p align="center">
  <img src="./assets/banner.svg" alt="Antigravity Kaggle Arena Banner" width="100%"/>
</p>

<h1 align="center">🚀 Antigravity Kaggle Arena</h1>

<p align="center">
  <strong>Autonomous Agentic Competitive Machine Learning Framework powered by Google DeepMind's Antigravity</strong>
</p>

<p align="center">
  <a href="https://github.com/topics/kaggle"><img src="https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle"/></a>
  <a href="https://github.com/topics/machine-learning"><img src="https://img.shields.io/badge/ML-GBDT%20%2B%20Neural-FF6F00?logo=scikitlearn&logoColor=white" alt="Machine Learning"/></a>
  <a href="https://github.com/topics/deep-learning"><img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://github.com/topics/agentic-ai"><img src="https://img.shields.io/badge/Framework-Antigravity%20Agentic-4285F4?logo=google&logoColor=white" alt="Antigravity"/></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"/></a>
  <a href="https://github.com/python/cpython"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue?logo=python&logoColor=white" alt="Python Version"/></a>
</p>

---

## 📌 Overview

**Antigravity Kaggle Arena** is an autonomous, self-improving machine learning framework designed to participate, iterate, and achieve **Gold-Medal / #1 World Leaderboard** standards across Kaggle competitions.

By combining Google DeepMind's **Antigravity Agentic Pair Programmer** with specialized community skills ([`nvidia-kaggle`](https://github.com/NVIDIA/nvidia-kaggle) by NVIDIA and [`scientific-agent-skills`](https://github.com/K-Dense-AI/scientific-agent-skills) by K-Dense-AI), the arena automates the complete competitive lifecycle:
1. **Competition Intelligence**: Autonomous retrieval and semantic distillation of official metrics, rules, and Grandmaster solution writeups via [`nvidia-kaggle-skill`](https://github.com/NVIDIA/nvidia-kaggle).
2. **Domain & Scientific Discovery**: Statistical, clinical, and tabular feature exploration powered by the extensive library of [`scientific-agent-skills`](https://github.com/K-Dense-AI/scientific-agent-skills).
3. **Leakage-Free Validation First**: Strict in-loop target encoding and stratified K-Fold schemas.
4. **Advanced Feature Engineering**: Multi-scale continuous binning, digit modulo residuals, domain physiological ratios, and group Z-score aggregations.
5. **Multi-Family Heterogeneous Modeling**: Ensembles combining asymmetric trees (LightGBM), symmetric oblivious trees (CatBoost), exact depth-wise trees (XGBoost), and Deep Residual Tabular Networks (PyTorch).
6. **Continuous Self-Improvement**: Automated logging of architectural lessons into an evolutionary knowledge base for subsequent challenges.

---

## 🏆 Arena Leaderboard & Competition Tracker

| Competition (Common Name) | Kaggle Link | Track / Domain | Metric | Date Achieved | Arena CV Score | Official Kaggle Score (Private / Public) | Benchmark World LB | Status | Detailed Solution Report |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Predicting Heart Disease** (`playground-series-s6e2`) | [View on Kaggle 🔗](https://www.kaggle.com/competitions/playground-series-s6e2) | Tabular / Clinical | **ROC-AUC** | **Sep 27, 2026** | **`0.95536`** | **`0.95496`** / **`0.95349`** *(Ref: 56613806)* | `0.95535` | 🥇 **#1 Benchmark Surpassed** | [Full Report & Walkthrough](docs/competitions/playground-series-s6e2.md) |

---

## 🏗️ Architecture & Operating Flow

```mermaid
flowchart TB
    subgraph S1["1. INTELLIGENCE & DOMAIN DISCOVERY"]
        direction LR
        KAG["Kaggle Competition Target"] --> INTEL["nvidia-kaggle-skill\n• Rules & Constraints\n• Winner Writeups (Masaya #1)"]
        SCI["K-Dense Scientific Skills\n• Clinical / Domain Priors\n• Statistical Explorations"] --> INTEL
    end

    subgraph S2["2. RIGOROUS VALIDATION FOUNDATION"]
        direction LR
        INTEL --> SCAFF["Scaffold Competition\nWorkspace Layout"]
        SCAFF --> SPLIT["Deterministic Folds (5-Fold Stratified)\n• Strict Train/Val Partitioning\n• Frozen Fold IDs"]
    end

    subgraph S3["3. ZERO-LEAKAGE FEATURE PIPELINE"]
        direction TB
        FE_ENG["Feature Matrix V2 (115+ Features)\n• Modulo Remainder Residuals (col % 10, % 5)\n• Multi-Scale Quantile & Uniform Binning\n• Domain Formulas (Tanaka Cardiac Reserve, BP/Chol)"]
        TE_LOOP["In-Loop Bayesian Target Encoding\n(Strictly fitted on Train Folds only)"]
        Z_SCORE["Group Normalized Z-Scores\n(Relative Deviations by Category)"]
        FE_ENG --> TE_LOOP --> Z_SCORE
    end

    subgraph S4["4. HETEROGENEOUS MULTI-MODEL STACK"]
        direction LR
        M1["CatBoost (Multi-Seed 42, 1337)\nSymmetric Oblivious Trees"]
        M2["LightGBM (Multi-Seed 42, 1337)\nLeaf-wise Asymmetric Trees"]
        M3["XGBoost\nExact Gradient Boosting"]
        M4["TabularResMLP (PyTorch)\nLayerNorm + SiLU + Skip Connections"]
    end

    subgraph S5["5. META-ENSEMBLING & SANITY GATE"]
        direction LR
        OPT["Optuna Bayesian Optimization\n• Percentile Rank Blending\n• 81.6% CatBoost + 18.4% LightGBM"]
        GATE["verify_submission.py\n• Zero NaN / Inf Check\n• Row Alignment Check\n• Probability Bounds [0, 1]"]
        OPT --> GATE
    end

    subgraph S6["6. CONTINUOUS EVOLUTIONARY LEARNING"]
        direction LR
        GATE --> SUBMIT["Final Scored Submission\n(Target: 0.95536)"]
        SUBMIT --> LOG["Evolution Knowledge Base\n(EVOLUTION_LOG.md)\n• Retrospective Auditing\n• Autonomous Skill Upgrades"]
    end

    SPLIT --> FE_ENG
    Z_SCORE --> S4
    S4 --> OPT
```

---

## 📂 Repository Structure

```text
.
├── README.md                          # Repository hub & master overview
├── LICENSE                            # Apache 2.0 Open Source License
├── requirements.txt                   # Production environment dependencies
├── docs/
│   ├── wiki/
│   │   ├── 01-mission-and-architecture.md  # Core mission, agentic loop & tooling
│   │   ├── 02-skills-and-modules.md        # Technical breakdown of custom skills
│   │   └── 03-evolution-playbook.md        # Lessons learned & self-improvement logs
│   └── competitions/
│       └── playground-series-s6e2.md       # Full end-to-end benchmark walkthrough
└── src/
    ├── core/
    │   ├── feature_engineering_v2.py       # Multi-scale binning, modulo & Z-scores
    │   ├── in_loop_target_encoder.py       # Strict leakage-free Bayesian target encoder
    │   ├── tabular_mlp.py                  # PyTorch Tabular Residual Neural Network
    │   ├── ridge_ensemble.py               # Ridge stacking & rank averaging module
    │   └── verify_submission.py            # Automated submission validator & sanity gate
    └── competitions/
        └── playground_s6e2/
            ├── train_v2_multiseed.py       # Winning multi-seed training pipeline
            └── train_grandmaster_stack.py  # 4-family heterogeneous ensemble runner
```

---

## 🚀 Quickstart & Reproduction

### 1. Prerequisites
- Python 3.10+ (tested up to Python 3.14 on macOS Apple Silicon and Linux).
- Kaggle API credentials (save token to `~/.kaggle/access_token` or export `KAGGLE_API_TOKEN`).

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/antigravity-kaggle-arena.git
cd antigravity-kaggle-arena

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Winning Playground S6E2 Pipeline
```bash
# Run feature engineering, 5-fold multi-seed training and Optuna blending
python src/competitions/playground_s6e2/train_v2_multiseed.py
```

---

## 📖 Documentation & Wiki
- 📘 [Mission, Architecture & Agentic Workflow](docs/wiki/01-mission-and-architecture.md)
- 🛠️ [Skills & Core Modules Specification](docs/wiki/02-skills-and-modules.md)
- 🧠 [Self-Improvement & Evolution Playbook](docs/wiki/03-evolution-playbook.md)
- 🔬 [Playground Series S6E2: Zero to #1 Walkthrough](docs/competitions/playground-series-s6e2.md)

---

## 📄 License
This project is open-source under the [Apache License 2.0](LICENSE).
