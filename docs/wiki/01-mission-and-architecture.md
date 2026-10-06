# Mission and operating architecture

## Mission

The Arena converts a user-defined competitive objective into a traceable sequence of experiments. It survives changes in competition, agent, model family, and accelerator: Codex and Antigravity are participants in a stable evidence loop, not the identity of the project.

## Responsibility model

| Participant | Responsibility |
|---|---|
| Human competitor | Sets the objective, resolves strategic choices, enables account resources, and approves submissions/publication |
| Codex or Antigravity | Inspects evidence, proposes hypotheses, edits and tests code, operates remote runs, and reports uncertainty |
| Specialist skills | Supply procedures for Kaggle operations, EDA, statistics, validation, and reporting |
| Kaggle | Provides datasets, CPU/GPU/TPU runtimes, submissions, and leaderboard evidence |
| GitHub | Preserves source, experiment history, audits, and the public Wiki |

```mermaid
flowchart TD
    U[Human objective] --> I[Agent inspects evidence]
    I --> H[One falsifiable hypothesis]
    H --> R[Implement and test restartable runner]
    R --> K[Kaggle CPU / GPU / TPU]
    K --> A[Aligned OOF and test artifacts]
    A --> C[Cross-fitted comparison]
    C --> D{Stable marginal gain?}
    D -- No --> X[Record rejection]
    D -- Yes --> G[Integrity and leakage gate]
    G --> P{Human approves submission?}
    P -- No --> W[Retain locally]
    P -- Yes --> S[Submit and label LB evidence]
    S --> N[Competition-specific report]
    X --> H
    N --> H
```

## Non-negotiable boundaries

1. Freeze validation before serious model search.
2. Fit supervised preprocessing only inside the active training partition.
3. Label every score as OOF, Public LB, or Private LB.
4. Judge ensemble additions by marginal held-out value.
5. Store IDs, folds, predictions, metrics, and restartable checkpoints.
6. Keep remote training separate from submission.
7. Publish important failures and limitations.

## Compute architecture

Local workspaces hold code, tests, folds, evaluation, and reports. Kaggle CPU runs long tree or linear workloads; Kaggle GPU supports neural and accelerated boosting workloads; TPU is selected only when the implementation can use it. GitHub synchronizes the version-controlled Wiki. This lets training continue when the local computer is closed and lets a future agent reconstruct every decision.
