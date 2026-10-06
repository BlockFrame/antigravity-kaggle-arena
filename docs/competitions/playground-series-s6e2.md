# Playground Series S6E2: complete Codex-assisted solution report

## Executive result

The project matched the displayed winning Private ROC-AUC for Kaggle
[Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2).

| Final evidence | Value |
|---|---:|
| Model | V17 cross-fitted rank blend |
| OOF ROC-AUC | **`0.955754551`** |
| Public leaderboard | **`0.95394`** |
| Private leaderboard | **`0.95535`** |
| Displayed winning Private score | **`0.95535`** |
| Submission reference | **`56862032`** |
| Submitted | 2026-10-05 21:20 UTC |

This was a late submission after the competition deadline. It demonstrates
winner-level model quality on the released scoring system but does not change
the historical official leaderboard or confer an official first-place finish.

## Starting point: audit before optimization

The first important change was not a model. It was correcting the evidence.
Earlier project material mixed three quantities that are not interchangeable:

- local or combined-data CV;
- Kaggle Public leaderboard score;
- Kaggle Private leaderboard score.

The best verified starting submission was V5 (`56651832`) at `0.95361` Public
and `0.95507` Private. The target was `0.95535` Private, leaving a real gap of
`0.00028`. From that point onward, OOF, Public, and Private were reported in
separate columns.

The competition contained 630,000 synthetic training rows. A separate original
heart-disease dataset contained 303 records; after target availability and
training-side filtering, 270 rows were used as the source for external
statistics in the relevant models.

## Verified Kaggle submission history

This table comes from the Kaggle submission history, not from local estimates.

| Ref | Candidate | Public | Private | Decision |
|---:|---|---:|---:|---|
| `56613806` | V2 multi-seed GBDT | `0.95349` | `0.95496` | Historical baseline |
| `56619981` | V3 feature-expanded stack | `0.95353` | `0.95502` | Improved |
| `56620791` | V4 full-data multi-seed tri-stack | `0.95359` | `0.95508` | Improved |
| `56651824` | V5 hybrid GBDT + neural | `0.95360` | `0.95506` | Rejected |
| `56651832` | V5 pure shallow GBDT | `0.95361` | `0.95507` | Starting best |
| `56685579` | V7 three-way rank blend | `0.95391` | `0.95532` | Major breakthrough |
| `56831251` | V14 bin/digit RealMLP blend | `0.95392` | `0.95534` | One point from target |
| `56833796` | V16 equal-seed blend | `0.95392` | `0.95534` | Stable but no LB gain |
| `56834230` | V16 + V15 CatBoost | `0.95391` | `0.95533` | Rejected after LB regression |
| **`56862032`** | **V17 final** | **`0.95394`** | **`0.95535`** | **Winner score matched** |

## Experimental path from V6 to V17

All values below are aligned OOF ROC-AUC unless a leaderboard score is
explicitly identified.

| Version | Candidate or ensemble | OOF | Outcome |
|---|---|---:|---|
| V6 | 50/50 rank blend: reconstructed V5 + OHE logistic | `0.955615645` | Accepted as diverse base |
| V7 | RealMLP all-categorical, seed 42 | `0.955702707` | Strong neural candidate |
| V7 | V6 + RealMLP + Ordered CatBoost D3 | `0.955733952` | Private `0.95532` |
| V8 | RealMLP hybrid numeric/categorical | `0.955521032` | Rejected; representation hurt |
| V9 | RealMLP all-categorical, seed 1337 | `0.955697605` | Useful stability check, not stronger |
| V10 | Ordered CatBoost D2 all-categorical | `0.955548399` | Rejected standalone |
| V11 | RealMLP all-categorical, `n_ens=20` | `0.955703832` | Tiny gain, poor compute return |
| V12 | Ordered CatBoost D3 hybrid | `0.955545045` | Rejected |
| V13 | RealMLP raw-only all-categorical | `0.955709517` | Accepted; simpler was better |
| V13 | Updated three-way blend | `0.955739753` | New base |
| V14 | RealMLP raw + bin/digit | `0.955715856` | Strong complementary candidate |
| V14 | V13 + 30% V14 | `0.955746123` | Private `0.95534` |
| V15 | CatBoost D2 raw + bin/digit | `0.955583754` | Locally diverse |
| V15 blend | V16 + 10% V15 | `0.955750482` | Private fell to `0.95533`; rejected |
| V16 | RealMLP bin/digit, seed 1337 | `0.955706227` | Seed stability check |
| V16 blend | 70% V13 + 15% each bin/digit seed | `0.955749097` | Private stayed `0.95534` |
| V17 | RealMLP bin/digit + original singletons | `0.955739108` | Best standalone new candidate |
| **V17 final** | **58% V14 + 42% V17** | **`0.955754551`** | **Private `0.95535`** |

### Phase 1 — Build a trustworthy validation boundary

Five stratified folds with seed 42 were frozen and reused. Each training runner
wrote:

- sample IDs;
- targets;
- fold assignments;
- OOF predictions;
- test predictions;
- metrics and fold timings;
- restartable fold checkpoints.

Alignment checks rejected arrays with different IDs, targets, folds, lengths,
or non-finite predictions. This mattered because a gain of `0.00001` is smaller
than the error created by even a subtle alignment or leakage defect.

### Phase 2 — V6 established representation diversity

The reconstructed V5 tree-family OOF scored `0.955430235`. An OHE logistic
model scored `0.955447125`. Neither was competitive alone, but their Spearman
correlation was `0.99662`, sufficiently different for a 50/50 rank blend to
reach `0.955615645`.

Why rank space? ROC-AUC depends on ordering rather than probability calibration.
Percentile ranks prevent one model's more extreme probabilities from dominating
the combination for the wrong reason.

### Phase 3 — V7 introduced the decisive neural family

RealMLP treated all 13 base columns as categorical and combined them with
statistics learned from the original source. Configuration:

- five folds, seed 42;
- `n_cv=2`, `n_ens=8`;
- 100 epochs, batch size 128;
- CUDA on Kaggle;
- 81 total features, including 68 original-stat features.

Standalone OOF reached `0.955702707`. A staged three-way rank blend with V6 and
Ordered CatBoost depth 3 achieved honest cross-fitted OOF `0.955733952`.
Candidate CatBoost weight was selected on four folds and evaluated on the held
out fifth. The deployed weights were 22.5% V6, 52.5% RealMLP, and 25% CatBoost.

Kaggle submission `56685579` jumped to `0.95532` Private, shrinking the target
gap from `0.00028` to `0.00003`.

### Phase 4 — GPU provisioning became part of the experiment

The first remote notebooks requested a T4 but actually received a CPU image:
PyTorch reported `2.10.0+cpu`, CUDA was false, and `nvidia-smi` was absent.
Changing model code could not fix infrastructure that had not provisioned a GPU.

The user completed Kaggle account verification and manually restarted the
session with GPU T4 x2 enabled. A fail-fast CUDA probe was added so future
RealMLP jobs would terminate instead of silently consuming hours on CPU.
Long-running work then moved off the local Mac:

- RealMLP variants ran on Kaggle GPU;
- CatBoost variants ran on Kaggle CPU;
- the local computer could be closed after the remote version entered running
  state;
- no training notebook submitted predictions automatically.

### Phase 5 — V8 through V13 searched for useful diversity

The search was deliberately diagnostic:

- **V8 hybrid representation failed.** Mixing numeric and categorical treatment
  reduced OOF to `0.955521032`.
- **V9 second seed did not outperform seed 42.** It measured stochastic
  stability but did not earn a large standalone weight.
- **V10 and V12 shallow CatBoost variants were weaker.** Shallow trees were
  retained only where they added ensemble diversity.
- **V11 increased `n_ens` from 8 to 20.** OOF improved by only `0.000001124`
  versus V7 while fold training time grew materially; scale was not the answer.
- **V13 removed all original statistics.** The 13-column raw categorical model
  improved to `0.955709517`, showing that the original pair statistics were
  adding noise as well as signal.

Replacing the earlier RealMLP with V13 in the three-way blend reached
`0.955739753` cross-fitted OOF.

### Phase 6 — V14 targeted synthetic generator artifacts

V14 expanded 13 raw columns to 48 categorical features with 35 engineered
representations:

- quantile bins;
- equal-width bins;
- rounded values;
- decimal digit and remainder features.

These features expose discontinuities and rounding patterns that a synthetic
data generator may preserve. The model reached `0.955715856` standalone. A 30%
candidate weight added to V13 produced `0.955746123` OOF and `0.95534` Private.

This was research guided by public winner write-ups, not a copy of the winning
pipeline. The winner used a much larger OOF library and a different selection
system. This project used a compact set of independently implemented models,
frozen folds, staged rank blends, and its own representation/statistics choices.

### Phase 7 — V15 and V16 prevented a premature conclusion

V15 asked whether CatBoost on the same bin/digit features supplied another
useful view. It did: a 10% blend raised local OOF from `0.955749097` to
`0.955750482`. Kaggle Private, however, fell from `0.95534` to `0.95533`.

That result changed the policy: tiny aggregate OOF gains were insufficient when
the candidate was highly correlated and fold evidence was weak. V15 was
discarded.

V16 averaged seed-42 and seed-1337 bin/digit models inside a pre-specified
70/15/15 blend. It improved OOF to `0.955749097`, but Private remained
`0.95534`. Seed averaging reduced variance without adding the missing signal.

### Phase 8 — V17 kept the original data signal and removed its noise

The final hypothesis combined the strongest V14 representation with a smaller,
cleaner external prior. V17 used:

- 48 raw/bin/digit categorical features;
- 52 original-source features;
- smoothed target mean, log-count, WoE, and entropy;
- singleton groups only;
- no original-source pair interactions;
- five folds, seed 42, `n_cv=2`, `n_ens=8`, 100 epochs, batch size 128.

V17 reached `0.955739108` standalone, substantially stronger than V14 and close
to the full prior ensemble. Against the fixed V14 base, the candidate weight
was selected outside each held-out fold:

| Held-out fold | Selected V17 weight |
|---:|---:|
| 0 | `0.42` |
| 1 | `0.37` |
| 2 | `0.51` |
| 3 | `0.42` |
| 4 | `0.40` |

Four of five held-out deltas were positive. The median deployment weight was
42%, yielding:

```text
final = 0.58 × rank(V14) + 0.42 × rank(V17)
OOF   = 0.955754551
delta = +0.000008428 versus V14
```

Submission `56862032` scored `0.95394` Public and `0.95535` Private.

## Final ensemble expanded

V14 was itself 70% V13 plus 30% V14 standalone; V13 was the updated V7
three-way blend. Expanding the nested rank ensemble gives:

| Component | Effective final weight |
|---|---:|
| V6 OHE/tree rank blend | `9.135%` |
| RealMLP raw all-categorical | `21.315%` |
| Ordered CatBoost depth 3 | `10.150%` |
| RealMLP raw + bin/digit | `17.400%` |
| RealMLP bin/digit + original singleton stats | **`42.000%`** |

These are weights over rank-transformed predictions. They should not be
interpreted as calibrated probability contributions.

## Technical choices and why they were made

| Choice | Reason |
|---|---|
| One frozen five-fold split | Makes every candidate directly comparable and prevents split shopping |
| External original data only for statistics | Transfers real-data priors without mixing validation labels into feature construction |
| All-categorical RealMLP | Matched the discrete/synthetic structure better than the hybrid representation |
| Shallow CatBoost | Added tree diversity while limiting memorization of synthetic artifacts |
| Bin/digit features | Exposed quantization and rounding structure hidden by raw continuous values |
| Singleton original statistics | Kept robust marginal priors; pair statistics were too sparse/noisy |
| Rank blending | Aligned combination with ROC-AUC and neutralized calibration-scale differences |
| Cross-fitted weight selection | Measured blend selection outside the fold being scored |
| Separate training and submission | Prevented accidental quota use and required explicit human approval |
| Checkpoints per fold | Made multi-hour Kaggle jobs restartable and auditable |
| Fail-fast GPU test | Prevented an incorrectly provisioned CPU session from wasting training time |

## What failed and what it taught us

1. **Documentation can overfit too.** A local `0.95536` was once described as
   if it were a Kaggle result. The audit removed that claim.
2. **More features are not automatically better.** Hybrid representation and
   original pair statistics both underperformed simpler alternatives.
3. **More ensemble members are not automatically better.** `n_ens=20` delivered
   negligible value over `n_ens=8` for much more compute.
4. **Multi-seed stability is not new signal.** V16 improved OOF but did not move
   the rounded leaderboard score.
5. **A local micro-gain can be false confidence.** V15 improved OOF and hurt
   Private LB; it was removed.
6. **Infrastructure state must be measured.** A metadata GPU flag did not mean
   CUDA was actually available.

## Human–Codex collaboration and prompting

The user did not provide a single exhaustive prompt. The process used a stable
goal and many short control prompts.

### The user's role

- Set an unambiguous terminal objective: continue until the winning score was
  reached or exceeded.
- Used short commands—“vai”, “prosegui”, “verifica”—to maintain execution
  momentum without micromanaging implementation.
- Asked for time estimates and status checks before long CPU/GPU runs.
- Asked what the fallback would be if an experiment failed.
- Challenged whether the approach was genuinely different from the winners.
- Enabled T4 x2 in Kaggle and resolved the account/session provisioning issue.
- Explicitly approved submission and publication actions.

### Codex's role

- Converted the objective into measurable experiment gates.
- Inspected repositories, Kaggle status, logs, metrics, and OOF artifacts.
- Implemented restartable runners and alignment-safe evaluators.
- Chose the next candidate from marginal value rather than standalone score.
- Kept training remote so the user's computer was not occupied.
- Reported uncertainty, rejected regressions, and stopped automatic submission.
- Updated code and documentation only after verified outcomes.

### Why the prompting pattern worked

The user controlled **direction and permission**, while Codex controlled
**implementation detail and evidence collection**. The prompts stayed compact
because the shared state—goal, best score, active runs, next decision—was
maintained across turns. Questions such as “quanto manca?”, “si è bloccato?”,
and “quanto siamo lontani dal top?” forced operational observability, while
“se non funziona cosa altro possiamo fare?” forced a live fallback roadmap.

This was a persistent human-agent loop, not one-shot prompt engineering.

## Skills invoked

The repository received 166 project-scoped K-Dense Scientific Agent Skills and
the NVIDIA Kaggle skill. Only the subset below was invoked for this result:

| Skill | Contribution |
|---|---|
| Codex `skill-installer` | Installed GitHub-hosted skills into `.agents/skills/` and verified every `SKILL.md` |
| NVIDIA `nvidia-kaggle-skill` | Competition inspection, notebook metadata, polling, artifact download, quota checks, anti-duplicate submission flow |
| K-Dense `exploratory-data-analysis` | Aggregate-safe EDA and synthetic/original dataset audit |
| K-Dense `scikit-learn` | Stratified folds, ROC-AUC evaluation, OOF protocol |
| K-Dense `statistical-analysis` | Fold deltas, stability assessment, conservative acceptance of micro-gains |
| Browser control | Inspected signed-in Kaggle session state during GPU diagnosis |
| `openai-docs` | Codex and Headroom configuration checks; unrelated to model scoring |
| `imagegen` | Produced an earlier banner concept, later replaced by a code-native SVG; it did not affect modeling |

Method provenance for the K-Dense procedures:

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: Library of Procedural Knowledge for Research Agents*.
arXiv:2609.00065. <https://doi.org/10.48550/arXiv.2609.00065>

## Codex token accounting

The final counter recorded immediately after the initial V17 GitHub publication
on 5 October 2026 was:

| Runtime counter | Tokens |
|---|---:|
| Total processed | **`82,233,374`** |
| Input | `81,945,846` |
| Cached input | `77,965,568` |
| Non-cached input | `3,980,278` |
| Output | `287,528` |
| Reasoning output | `91,760` |

Cached input represented `95.14%` of input processing. Reasoning output is a
subset of output rather than an additional amount.

The counter covers the continuous Codex project session from 28 September to
the first publication, including system context, repeated cached conversation,
and tool outputs. It is not a count of unique prose, a per-model-training token
cost, or a directly billable total. Headroom's ledger covered other Codex work
as well, so no Headroom savings number is attributed to this competition.

## Reproduction entry points

- `src/competitions/playground_s6e2/train_v7_realmlp.py`
- `src/competitions/playground_s6e2/train_v7_catboost_ordered.py`
- `src/competitions/playground_s6e2/evaluate_oof_candidate.py`
- `src/competitions/playground_s6e2/run_v15_catboost_d2_bin_digit_cpu.py`
- `src/competitions/playground_s6e2/run_v16_realmlp_bin_digit_seed1337.py`
- `src/competitions/playground_s6e2/run_v17_realmlp_bin_digit_orig_singletons.py`

The final runner expects Kaggle-mounted competition data, the original dataset,
and the pytabkit model package. Submission is intentionally not performed by
the runner.

## Final conclusion

The winning point did not come from brute-forcing more versions of the same
model. It came from progressively improving the evidence system:

- correct score accounting;
- one immutable validation boundary;
- representation diversity;
- remote, restartable compute;
- held-out ensemble selection;
- explicit rejection of attractive failures;
- and a final feature set that retained robust original-data priors while
  deleting noisy interactions.

That process moved the verified Private score from `0.95507` to **`0.95535`**.
