# Wiki: Project Mission & Agentic Architecture

## 1. Executive Summary & Objective

The primary objective of the **Antigravity Kaggle Arena** is to pioneer an autonomous, rigorous, and continuously evolving competitive machine learning system using **Google DeepMind's Antigravity AI coding assistant**.

Competitive data science on Kaggle typically suffers from fragmented workflows:
- Feature generation is often ad-hoc and prone to target leakage.
- Ensembling frequently falls into the trap of stacking collinear models without true statistical diversity.
- Learnings from one competition are rarely formalized into programmatic tools for the next challenge.

This project bridges these gaps by establishing an **autonomous agentic pair programming framework** that treats every Kaggle competition as a validation-first problem, extracts intelligence from top Grandmasters, constructs robust multi-model pipelines, and archives evolutionary breakthroughs.

---

## 2. Integrated Tooling & Skills Ecosystem

The system synthesizes two cutting-edge agentic toolkits:

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

## 3. The 5-Stage Agentic Operating Loop

Every competition entered by the Arena follows a strict 5-stage lifecycle:

### Stage 1: Intelligence Discovery & Metric Extraction
- Queries the Kaggle API to inspect the official scoring metric, data dictionary, submission format, and hidden-test constraints.
- Employs `fetch_leaderboard_writeups.py` and `fetch_writeup.py` to harvest insights, validation splits, and structural nuances from past winners.

### Stage 2: Validation-First Architecture & Scaffolding
- Instantiates a clean, standardized project layout (`input/`, `src/`, `models/`, `oof/`, `submissions/`).
- Freezes a deterministic Stratified K-Fold / Group K-Fold split column before any feature engineering begins.

### Stage 3: Feature Engineering with Zero Leakage
- Generates multi-scale representations (binning, digit modulo residuals, domain physiological formulas, group Z-scores).
- All statistics and target encodings are calculated **strictly within the training folds** of each split.

### Stage 4: Heterogeneous Multi-Family Modeling & Blending
- Trains diverse model families (CatBoost, LightGBM, XGBoost, and PyTorch Tabular ResMLP).
- Executes multi-seed averaging per fold to suppress stochastic split variance.
- Applies Bayesian optimization via Optuna to determine optimal rank-averaging weights.

### Stage 5: Verification Gate & Evolutionary Feedback
- Runs `verify_submission.py` to guarantee non-null values, row integrity, column alignment, and valid probability boundaries.
- Records all performance deltas, correlation metrics, and architectural decisions into `EVOLUTION_LOG.md`.
