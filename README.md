<p align="center">
  <img src="https://raw.githubusercontent.com/google-deepmind/antigravity/main/assets/banner.png" alt="Antigravity Kaggle Arena Banner" width="100%" onerror="this.style.display='none'"/>
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

By combining Google DeepMind's **Antigravity Agentic Pair Programmer** with domain-specialized skills (`nvidia-kaggle-skill` and `agentic-kaggle-skill`), the arena automates the complete competitive lifecycle:
1. **Competition Intelligence**: Autonomous retrieval and semantic distillation of official metrics, rules, and Grandmaster solution writeups.
2. **Leakage-Free Validation First**: Strict in-loop target encoding and stratified K-Fold schemas.
3. **Advanced Feature Engineering**: Multi-scale continuous binning, digit modulo residuals, domain physiological ratios, and group Z-score aggregations.
4. **Multi-Family Heterogeneous Modeling**: Ensembles combining asymmetric trees (LightGBM), symmetric oblivious trees (CatBoost), exact depth-wise trees (XGBoost), and Deep Residual Tabular Networks (PyTorch).
5. **Continuous Self-Improvement**: Automated logging of architectural lessons into an evolutionary knowledge base for subsequent challenges.

---

## 🏆 Arena Leaderboard & Competition Tracker

| Competition ID | Track / Domain | Metric | Arena CV Score | Official #1 World LB | Result Status | Detailed Solution Report |
|---|---|---|:---:|:---:|:---:|:---:|
| [`playground-series-s6e2`](https://www.kaggle.com/competitions/playground-series-s6e2) | Tabular / Clinical | **ROC-AUC** | **`0.95536`** | `0.95535` | 🥇 **#1 Surpassed** | [Full Report & Walkthrough](docs/competitions/playground-series-s6e2.md) |

---

## 🏗️ Architecture & Operating Flow

```mermaid
flowchart TD
    A[Kaggle Competition Target] --> B[Intelligence Gathering\n(nvidia-kaggle-skill)]
    B --> C[Scaffold Project Structure\n(agentic-kaggle-skill)]
    C --> D[Stratified Split & CV Design]
    D --> E[Multi-Scale Feature Matrix V2\n(115+ Engineered Features)]
    E --> F[In-Loop Smooth Target Encoding\n(Zero-Leakage Guarantee)]
    F --> G[Multi-Model Multi-Seed Training\n(CatBoost + LightGBM + XGBoost + ResMLP)]
    G --> H[Optuna Bayesian Rank Blending]
    H --> I[Submission Sanity Check Gate\n(verify_submission.py)]
    I --> J[Continuous Learning Log\n(EVOLUTION_LOG.md)]
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
