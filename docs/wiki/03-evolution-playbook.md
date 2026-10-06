# Evolution playbook: V1 to V17

## Score progression

| Stage | What changed | Evidence |
|---|---|---|
| V1–V4 | Baselines, engineered GBDTs, multi-seed full training | Best early Private `0.95508` |
| V5 | Shallow GBDT family with original priors | Audited starting Private `0.95507` |
| V6 | OHE logistic diversity added in rank space | OOF `0.955615645` |
| V7 | All-categorical RealMLP + Ordered CatBoost | Private `0.95532` |
| V8–V12 | Representation, seed, depth, and ensemble-size controls | Mostly rejected |
| V13 | Raw-only RealMLP simplified noisy priors | Blend OOF `0.955739753` |
| V14 | Bin/digit representations | Private `0.95534` |
| V15 | CatBoost bin/digit stress test | Private regressed to `0.95533` |
| V16 | Pre-specified seed averaging | Private remained `0.95534` |
| V17 | Bin/digit + original singleton statistics | **Private `0.95535`** |

## Breakthroughs

### 1. Fix the measurement before the model

The original project narrative confused local CV with leaderboard evidence.
Once V5 was audited at `0.95507` Private, the real task became measurable.

### 2. Add a genuinely different representation

V6's logistic model was weaker alone but improved the tree blend. V7's
all-categorical RealMLP then supplied the largest new signal and moved Private
to `0.95532`.

### 3. Prefer simplification when it wins

Hybrid numeric treatment, pair statistics, and `n_ens=20` sounded stronger but
did not justify their complexity. Raw-only V13 was better than the original
feature-heavy RealMLP.

### 4. Target the generator without memorizing it

Quantile/equal-width bins, rounding, and digit/remainder features helped
RealMLP exploit stable synthetic structure. Shallow trees limited the tendency
to memorize fragile boundaries.

### 5. Separate robust priors from noisy interactions

V17 restored original-data statistics only for singleton groups. Target mean,
log-count, WoE, and entropy survived; sparse pair interactions did not.

### 6. Let failures update policy

V15's local micro-gain failed on Private LB. The response was not another
ad-hoc weight tweak: correlated micro-gains thereafter required stronger fold
evidence.

## Prompting playbook

The user maintained one durable objective and used compact control prompts:

- “continue until we reach or exceed first” fixed the terminal condition;
- “vai” and “prosegui” delegated the next evidence-backed action;
- “quanto manca?” and “si è bloccato?” enforced operational monitoring;
- “what else can we do if it fails?” forced fallback planning;
- “are we copying the winners?” forced provenance and originality analysis;
- explicit approval guarded submissions and publication.

This pattern is effective when the agent preserves state and exposes the next
decision. It is less effective if short prompts are interpreted without the
shared experiment history.

## Reusable rules

1. Label every score as OOF, Public, or Private.
2. Freeze folds before representation search.
3. Store IDs, targets, folds, OOF, test predictions, metrics, and checkpoints.
4. Judge candidates by marginal cross-fitted value, not standalone AUC.
5. For ranking metrics, test rank blends before raw probability blends.
6. Vary one major hypothesis per experiment.
7. Treat seed averaging as variance reduction, not automatic diversity.
8. Reject locally attractive candidates that fail external evidence.
9. Keep training remote and submission manual.
10. Publish limitations alongside the best score.

## Token transparency snapshot

From the initial Codex request on 28 September through the first V17 publication
on 5 October, the runtime counter reported `82,233,374` processed tokens:
`81,945,846` input, `77,965,568` cached input, and `287,528` output. Cached
input was `95.14%` of input. These counters include repeated context and tool
results and are not unique authored tokens or a direct cost figure.

For the complete technical and prompting history, read
[[Playground-Series-S6E2|the S6E2 report]].
