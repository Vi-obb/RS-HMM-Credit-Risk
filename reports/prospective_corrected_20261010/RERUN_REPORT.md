# Corrected credit-risk validation rerun — 10 October 2026

The pooled Brier-score and log-loss comparisons disagree in direction. The earlier blanket negative-result conclusion must be revised.

## Verified primary results

The three origins cover April 2021–March 2024: **7,023,684 scored loan-months** and **40,073 positive twelve-month labels**. Positive labels are not unique default events. Brier score and log loss are better when lower; ROC-AUC and average precision (AP) are better when higher.

The uncertainty intervals must qualify any point-ranking claim. In particular,
intervals containing zero do not establish a consistent proper-score advantage.
Mean predicted PD versus the observed rate also shows absolute calibration;
improved discrimination does not imply well-calibrated probabilities.

| Model | Period | N | Positive labels | Brier | Log loss | ROC-AUC | AP | Mean PD | Observed rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_logit | pooled | 7,023,684 | 40,073 | 0.005534 | 0.036582 | 0.726647 | 0.110008 | 0.016378 | 0.005705 |
| macro_logit | pooled | 7,023,684 | 40,073 | 0.021488 | 0.109906 | 0.524153 | 0.020207 | 0.077966 | 0.005705 |
| regime_aware_logit | pooled | 7,023,684 | 40,073 | 0.005568 | 0.036506 | 0.811853 | 0.144590 | 0.019613 | 0.005705 |

### Results by origin

| Model | Period | N | Positive labels | Brier | Log loss | ROC-AUC | AP | Mean PD | Observed rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_logit | origin_2021_04 | 1,998,215 | 9,944 | 0.005467 | 0.044814 | 0.531595 | 0.009763 | 0.025287 | 0.004976 |
| macro_logit | origin_2021_04 | 1,998,215 | 9,944 | 0.021643 | 0.100733 | 0.442784 | 0.005412 | 0.069180 | 0.004976 |
| regime_aware_logit | origin_2021_04 | 1,998,215 | 9,944 | 0.005338 | 0.040888 | 0.785696 | 0.107331 | 0.025687 | 0.004976 |
| baseline_logit | origin_2022_04 | 2,290,732 | 12,348 | 0.005067 | 0.033223 | 0.837256 | 0.154805 | 0.015851 | 0.005390 |
| macro_logit | origin_2022_04 | 2,290,732 | 12,348 | 0.039296 | 0.202481 | 0.710918 | 0.095004 | 0.173730 | 0.005390 |
| regime_aware_logit | origin_2022_04 | 2,290,732 | 12,348 | 0.005162 | 0.031713 | 0.851821 | 0.149621 | 0.015559 | 0.005390 |
| baseline_logit | origin_2023_04 | 2,734,737 | 17,781 | 0.005973 | 0.033380 | 0.807379 | 0.148070 | 0.010310 | 0.006502 |
| macro_logit | origin_2023_04 | 2,734,737 | 17,781 | 0.006458 | 0.039064 | 0.716303 | 0.062387 | 0.004170 | 0.006502 |
| regime_aware_logit | origin_2023_04 | 2,734,737 | 17,781 | 0.006077 | 0.037320 | 0.820164 | 0.173356 | 0.018572 | 0.006502 |

### Dependence-aware uncertainty

Differences below are regime-aware minus borrower-only, using 10,000 moving-block replicates. Three-month blocks are primary; 1/6/12-month blocks are prespecified sensitivity checks. These intervals condition on fixed fitted models and only 36 observation months. Bootstrap sign fractions are not null-hypothesis p-values. Repeated loans and overlapping twelve-month labels remain.

| metric | block_months | estimate_regime_minus_baseline | ci_lower_95 | ci_upper_95 |
| --- | --- | --- | --- | --- |
| brier | 1 | 0.000034898 | -0.000027211 | 0.000092158 |
| log_loss | 1 | -0.000075475 | -0.001267667 | 0.001054227 |
| brier | 3 | 0.000034898 | -0.000069610 | 0.000116853 |
| log_loss | 3 | -0.000075475 | -0.002182209 | 0.001722803 |
| brier | 6 | 0.000034898 | -0.000097692 | 0.000121444 |
| log_loss | 6 | -0.000075475 | -0.002830234 | 0.002220309 |
| brier | 12 | 0.000034898 | -0.000042563 | 0.000110551 |
| log_loss | 12 | -0.000075475 | -0.002923512 | 0.002223248 |

## Corrected terminal comparison

The April 2023–September 2024 terminal holdout uses the same development windows and HMM cutoff as the last rolling origin. It changes only the test end date. On the shared 2,734,737 rows, exported model parameters and HMM paths match exactly. Scores agree numerically within 1e-8, with maximum probability differences {'baseline_logit': 0.0, 'macro_logit': 0.0, 'regime_aware_logit': 3.0900702419067727e-07}. The small regime-only scoring difference is consistent with float32 interaction/inference effects when scoring different batch sizes; this is numerical agreement, not byte reproduction. Detailed metric differences are recorded in terminal_overlap_verification.json. The original full-window terminal HMM results are historical diagnostics and are not promoted.

| Model | Period | N | Positive labels | Brier | Log loss | ROC-AUC | AP | Mean PD | Observed rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_logit | pooled | 4,260,526 | 29,180 | 0.006281 | 0.035191 | 0.807727 | 0.153761 | 0.011883 | 0.006849 |
| macro_logit | pooled | 4,260,526 | 29,180 | 0.006803 | 0.040971 | 0.739751 | 0.052574 | 0.004073 | 0.006849 |
| regime_aware_logit | pooled | 4,260,526 | 29,180 | 0.006434 | 0.039577 | 0.821783 | 0.180632 | 0.020484 | 0.006849 |

Terminal bootstrap sensitivity is recorded separately in ../prospective_terminal_20261010/bootstrap_block_sensitivity.csv. Its 18 months provide less temporal information than the primary comparison.

## Information rules and population

- Predict at the start of month t+1 from the month-end t servicing state. This assumes internal lender servicing availability; historical public Freddie files do not prove the real-time availability of every borrower field.
- The outcome spans t+1 through t+12. Supervised labels become available uniformly at t+14, allowing an additional reporting month. All training, tuning and calibration labels mature strictly before test issuance; the three development windows are disjoint. Development scores are retrospective diagnostics, not historical issued forecasts.
- For the first origin: train through February 2018, tune March 2018–February 2019, calibrate March 2019–February 2020, test April 2021–March 2022. Move these boundaries one year for each next origin. The HMM, scaling, factor statistics and imputers fit only through the training cutoff.
- Macroeconomic features use ALFRED vintages at month-end t, before the target starts. Retain the latest complete common economic month available within that vintage, without future revisions. Across 129 macro months, economic observation lags are {'2': 128, '3': 1} months. This deliberately sacrifices fresher fast-release indicators for a consistent common vector. Raw downloads and hashes are cached locally. [ALFRED vintage documentation](https://alfred.stlouisfed.org/help/downloaddata)
- First observed 90+ DPD before voluntary payoff resolves a positive outcome. Voluntary payoff (code 01) resolves a competing negative outcome. Other exits, gaps, missing delinquency status and insufficient follow-up remain censored. Unknown feature lookbacks exclude the prediction row. Continuous servicing is required only through resolution, rather than survival to horizon end.
- The input through September 2024 contains 17,625,857 eligible loan-months, 481,804 loans and 184,558 positive labels. The primary observation span through March 2024 contains 16,100,068 rows. In the 36 test months, 3,316 at-risk rows have unresolved outcomes and are excluded. Monthly coverage and payoff counts are in outcome_coverage.csv. Informative censoring is not proved absent; this is not validation over every active loan.
- Keep the same three model specifications, alpha candidates, seeds and primary Platt map. Raise SGD's cap uniformly from 30 to 300 after development fits hit the old cap, before test scores were assessed. Every tuning candidate and final classifier stopped before the cap; recorded iterations range from 8 to 76. This is an algorithmic stopping check, not proof of a global optimum.

Several information/population corrections change the experiment together. Do not attribute a ranking difference to a single correction, compare old and new score magnitudes as like-for-like, or describe the terminal/rolling contrast as isolating one validation effect.

## Verification, source and cleanup

All 49 repository tests passed. Independent verification recomputed fold and pooled Brier/log loss/ROC-AUC/AP from saved predictions, reconstructed paired monthly losses and all four bootstrap intervals, reconciled coverage, checked label maturity, and checked HMM seed selection and optimizer limits. See verification.json and terminal_overlap_verification.json; the terminal run has its own verification.json.

Primary fit commit: `ca8bbc49e47b9b9e531d5a0e14bc9cabba510ec6`. Terminal fit commit: `ca8bbc49e47b9b9e531d5a0e14bc9cabba510ec6`. Both ran on main in the frozen uv environment; manifests record package versions and input/output hashes. Test-level predictions remain local in ignored model_predictions.csv files; aggregate tables and provenance are suitable for Git synchronization.

All four secondary worktrees were removed after full file-by-file archive verification. Main already contained their code changes plus later corrections. Unique historical reproducibility evidence was recovered and pushed. Full archives and binary patches remain in ../RS-HMM-Credit-Risk-worktree-backups-20261010 outside the checkout. Older thesis/slide versions are preserved there; the unique local Scientific African manuscript was copied into the ignored manuscripts directory.

Stopped preliminary 30-iteration/delayed-issuance attempts are retained locally as diagnostics, excluded from the final evidence and Git synchronization. The final clock puts issuance before the target begins. No submission or journal transfer was made.

## Author-review implications

Use these new results and protocol to replace the numerical tables, abstract and discussion in the author-review manuscript. Review the target's voluntary-payoff treatment, servicing-availability assumption, residual censoring and limited temporal sample with the co-authors. The negative-result narrative must follow the corrected scores above. Existing manuscript documents have not been silently rewritten or submitted.

Reproduce with scripts/run_prospective.py, scripts/verify_prospective.py, and scripts/summarize_prospective.py. Prepare a fresh shared data directory with --terminal --prepare-only first, so both evaluations use inputs through September 2024; the runner reuses existing prepared inputs. Then run primary and --terminal evaluations against that same data directory. Protocol details and commands are in docs/prospective-rerun-20261010.md; exact input paths/hashes are in each run_manifest.json.
