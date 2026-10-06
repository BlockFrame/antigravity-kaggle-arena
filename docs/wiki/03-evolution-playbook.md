# Cross-competition evolution playbook

The Arena evolves through evidence accumulated across competitions. Detailed version histories belong to their case-study pages; this page contains the rules carried into the next challenge.

## Establish truth

- Audit data provenance, target, metric, and leakage risks.
- Reconcile documentation with actual OOF and leaderboard artifacts.
- Freeze folds, IDs, target order, and artifact schemas.
- Create a cheap reproducible baseline before expensive search.

## Search for complementary signal

- Change one major factor per experiment.
- Explore genuinely different representations and model families.
- Measure fold-level deltas and correlations against the current ensemble.
- Use cross-fitted or otherwise held-out blend selection.
- Prefer a small stable gain to a complex in-sample optimum.

## Operate remote compute safely

- Match workloads to Kaggle CPU, GPU, or TPU.
- Fail fast when the requested accelerator is unavailable.
- Save fold checkpoints and a run manifest.
- Poll and download artifacts without coupling training to a local laptop.
- Never let a training kernel submit automatically.

## External evidence and reporting

- Run an integrity gate before upload and submit only after human approval.
- Record score type, submission ID, date, and competition state.
- Keep leaderboard regressions in the experiment ledger.
- Publish one dedicated report per competition and feed reusable lessons back here.

## First case study

Playground Series S6E2 showed that measurement repair, representation diversity, generator-aware features, and cross-fitted rank blending mattered more than blindly increasing model size. Several plausible variants were rejected; V17 matched the displayed winning Private score. The full progression, prompting approach, skills, compute history, token snapshot, and limitations are in [[Playground-Series-S6E2|the dedicated S6E2 report]].
