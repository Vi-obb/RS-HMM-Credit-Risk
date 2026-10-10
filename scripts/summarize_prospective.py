from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.plot_rolling_results import plot_fold_scores, plot_fold_calibration, plot_monthly_differences


def markdown_table(frame: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows += ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def score_table(metrics: pd.DataFrame) -> str:
    rows = []
    for row in metrics.itertuples():
        rows.append({"Model": row.model, "Period": row.fold, "N": f"{row.n:,}", "Positive labels": f"{row.event_count:,}",
                     "Brier": f"{row.brier:.6f}", "Log loss": f"{row.log_loss:.6f}", "ROC-AUC": f"{row.roc_auc:.6f}",
                     "AP": f"{row.pr_auc:.6f}", "Mean PD": f"{row.pred_mean:.6f}", "Observed rate": f"{row.event_rate:.6f}"})
    return markdown_table(pd.DataFrame(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling-dir", type=Path, required=True)
    parser.add_argument("--terminal-dir", type=Path, required=True)
    args = parser.parse_args()
    r, t = args.rolling_dir, args.terminal_dir
    for directory in (r, t):
        assert json.loads((directory / "verification.json").read_text())["completed"]
    primary = pd.read_csv(r / "origin_2023_04/model_predictions.csv")
    terminal = pd.read_csv(t / "terminal_2023_04/model_predictions.csv")
    shared = terminal.loc[terminal.month_date.le("2024-03-01")].reset_index(drop=True)
    assert primary.panel_row.tolist() == shared.panel_row.tolist()
    assert primary.y_12m.tolist() == shared.y_12m.tolist()
    differences = {}
    metric_differences = {}
    for name, keys in [("rolling_origin_model_selection", ["model", "quantity"]),
                       ("rolling_origin_coefficients", ["model", "feature"]),
                       ("rolling_origin_hmm_initialisations", ["seed"]),
                       ("rolling_origin_hmm_paths", ["month_date"])]:
        left = pd.read_csv(r / (name + ".csv")); left = left.loc[left.fold.eq("origin_2023_04")].drop(columns="fold")
        right = pd.read_csv(t / (name + ".csv")).drop(columns="fold")
        if name.endswith("paths"):
            right = right.loc[right.month_date.le("2024-03-01")]
        pd.testing.assert_frame_equal(left.sort_values(keys).reset_index(drop=True), right.sort_values(keys).reset_index(drop=True), check_exact=True)
    y = primary.y_12m.to_numpy()
    for model in ("baseline_logit", "macro_logit", "regime_aware_logit"):
        np.testing.assert_allclose(primary[model], shared[model], rtol=0, atol=5e-7)
        differences[model] = float(np.max(np.abs(primary[model]-shared[model])))
        a, b = primary[model].to_numpy(), shared[model].to_numpy()
        metric_differences[model] = {"brier": float(np.mean((a-y)**2)-np.mean((b-y)**2)),
                                   "log_loss": float(np.mean(-y*np.log(a)-(1-y)*np.log1p(-a))-np.mean(-y*np.log(b)-(1-y)*np.log1p(-b))),
                                   "roc_auc": float(roc_auc_score(y,a)-roc_auc_score(y,b)),
                                   "ap": float(average_precision_score(y,a)-average_precision_score(y,b))}
        assert all(abs(value) < 1e-8 for value in metric_differences[model].values())
    (r / "terminal_overlap_verification.json").write_text(json.dumps({"n": len(primary), "exported_fitted_parameters_and_hmm_paths_identical": True,
            "shared_predictions_numerically_agree": True, "probability_absolute_tolerance": 5e-7,
            "metric_absolute_tolerance": 1e-8, "max_absolute_probability_difference": differences,
            "metric_differences": metric_differences,
            "note": "The small regime-only scoring difference is consistent with float32 interaction/inference effects for different test batch sizes; this is numerical rather than byte reproduction."}, indent=2))
    metrics = pd.read_csv(r / "rolling_origin_metrics.csv")
    terminal_metrics = pd.read_csv(t / "rolling_origin_metrics.csv")
    pooled = metrics.loc[metrics.fold.eq("pooled") & metrics.split.eq("test")]
    uncertainty = pd.read_csv(r / "rolling_origin_paired_uncertainty.csv")
    sensitivity = pd.read_csv(r / "bootstrap_block_sensitivity.csv")
    checks = json.loads((r / "panel_checks.json").read_text())
    manifest = json.loads((r / "run_manifest.json").read_text())
    terminal_manifest = json.loads((t / "run_manifest.json").read_text())
    selection = pd.read_csv(r / "rolling_origin_model_selection.csv")
    iterations = selection.loc[selection.quantity.str.endswith("iterations"), "value"]
    estimates = uncertainty.estimate_regime_minus_baseline
    if estimates.gt(0).all():
        finding = "The borrower-only model has lower pooled Brier score and log loss than the regime-aware model. The direction of the earlier negative result survives this corrected experiment."
    elif estimates.lt(0).all():
        finding = "The regime-aware model has lower pooled Brier score and log loss than the borrower-only model. The earlier negative-result conclusion must be revised."
    else:
        finding = "The pooled Brier-score and log-loss comparisons disagree in direction. The earlier blanket negative-result conclusion must be revised."
    plot_fold_scores(metrics, r / "figures/rolling_fold_scores.png")
    plot_fold_calibration(metrics, r / "figures/rolling_fold_calibration.png")
    plot_monthly_differences(pd.read_csv(r / "rolling_origin_paired_monthly_losses.csv"), r / "figures/rolling_monthly_loss_differences.png")
    bootstrap = sensitivity[["metric", "block_months", "estimate_regime_minus_baseline", "ci_lower_95", "ci_upper_95"]].copy()
    for column in bootstrap.columns[2:]:
        bootstrap[column] = bootstrap[column].map(lambda value: f"{value:.9f}")
    coverage = pd.read_csv(r / "outcome_coverage.csv")
    test_coverage = coverage.loc[coverage.month_date.between("2021-04-01", "2024-03-01")]
    report = f"""# Corrected credit-risk validation rerun — 10 October 2026

{finding}

## Verified primary results

The three origins cover April 2021–March 2024: **{int(pooled.n.iloc[0]):,} scored loan-months** and **{int(pooled.event_count.iloc[0]):,} positive twelve-month labels**. Positive labels are not unique default events. Brier score and log loss are better when lower; ROC-AUC and average precision (AP) are better when higher.

The uncertainty intervals must qualify any point-ranking claim. In particular,
intervals containing zero do not establish a consistent proper-score advantage.
Mean predicted PD versus the observed rate also shows absolute calibration;
improved discrimination does not imply well-calibrated probabilities.

{score_table(pooled)}

### Results by origin

{score_table(metrics.loc[metrics.split.eq('test') & ~metrics.fold.eq('pooled')])}

### Dependence-aware uncertainty

Differences below are regime-aware minus borrower-only, using 10,000 moving-block replicates. Three-month blocks are primary; 1/6/12-month blocks are prespecified sensitivity checks. These intervals condition on fixed fitted models and only 36 observation months. Bootstrap sign fractions are not null-hypothesis p-values. Repeated loans and overlapping twelve-month labels remain.

{markdown_table(bootstrap)}

## Corrected terminal comparison

The April 2023–September 2024 terminal holdout uses the same development windows and HMM cutoff as the last rolling origin. It changes only the test end date. On the shared {len(primary):,} rows, exported model parameters and HMM paths match exactly. Scores agree numerically within 1e-8, with maximum probability differences {differences}. The small regime-only scoring difference is consistent with float32 interaction/inference effects when scoring different batch sizes; this is numerical agreement, not byte reproduction. Detailed metric differences are recorded in terminal_overlap_verification.json. The original full-window terminal HMM results are historical diagnostics and are not promoted.

{score_table(terminal_metrics.loc[terminal_metrics.fold.eq('pooled')])}

Terminal bootstrap sensitivity is recorded separately in ../{t.name}/bootstrap_block_sensitivity.csv. Its 18 months provide less temporal information than the primary comparison.

## Information rules and population

- Predict at the start of month t+1 from the month-end t servicing state. This assumes internal lender servicing availability; historical public Freddie files do not prove the real-time availability of every borrower field.
- The outcome spans t+1 through t+12. Supervised labels become available uniformly at t+14, allowing an additional reporting month. All training, tuning and calibration labels mature strictly before test issuance; the three development windows are disjoint. Development scores are retrospective diagnostics, not historical issued forecasts.
- For the first origin: train through February 2018, tune March 2018–February 2019, calibrate March 2019–February 2020, test April 2021–March 2022. Move these boundaries one year for each next origin. The HMM, scaling, factor statistics and imputers fit only through the training cutoff.
- Macroeconomic features use ALFRED vintages at month-end t, before the target starts. Retain the latest complete common economic month available within that vintage, without future revisions. Across {checks['macro_months']} macro months, economic observation lags are {checks['macro_common_month_lag_counts']} months. This deliberately sacrifices fresher fast-release indicators for a consistent common vector. Raw downloads and hashes are cached locally. [ALFRED vintage documentation](https://alfred.stlouisfed.org/help/downloaddata)
- First observed 90+ DPD before voluntary payoff resolves a positive outcome. Voluntary payoff (code 01) resolves a competing negative outcome. Other exits, gaps, missing delinquency status and insufficient follow-up remain censored. Unknown feature lookbacks exclude the prediction row. Continuous servicing is required only through resolution, rather than survival to horizon end.
- The input through September 2024 contains {checks['rows']:,} eligible loan-months, {checks['loans']:,} loans and {checks['positive_labels']:,} positive labels. The primary observation span through March 2024 contains {checks['primary_observation_window_rows']:,} rows. In the 36 test months, {int(test_coverage.censored.sum()):,} at-risk rows have unresolved outcomes and are excluded. Monthly coverage and payoff counts are in outcome_coverage.csv. Informative censoring is not proved absent; this is not validation over every active loan.
- Keep the same three model specifications, alpha candidates, seeds and primary Platt map. Raise SGD's cap uniformly from 30 to 300 after development fits hit the old cap, before test scores were assessed. Every tuning candidate and final classifier stopped before the cap; recorded iterations range from {int(iterations.min())} to {int(iterations.max())}. This is an algorithmic stopping check, not proof of a global optimum.

Several information/population corrections change the experiment together. Do not attribute a ranking difference to a single correction, compare old and new score magnitudes as like-for-like, or describe the terminal/rolling contrast as isolating one validation effect.

## Verification, source and cleanup

All 49 repository tests passed. Independent verification recomputed fold and pooled Brier/log loss/ROC-AUC/AP from saved predictions, reconstructed paired monthly losses and all four bootstrap intervals, reconciled coverage, checked label maturity, and checked HMM seed selection and optimizer limits. See verification.json and terminal_overlap_verification.json; the terminal run has its own verification.json.

Primary fit commit: `{manifest['git_commit']}`. Terminal fit commit: `{terminal_manifest['git_commit']}`. Both ran on main in the frozen uv environment; manifests record package versions and input/output hashes. Test-level predictions remain local in ignored model_predictions.csv files; aggregate tables and provenance are suitable for Git synchronization.

All four secondary worktrees were removed after full file-by-file archive verification. Main already contained their code changes plus later corrections. Unique historical reproducibility evidence was recovered and pushed. Full archives and binary patches remain in ../RS-HMM-Credit-Risk-worktree-backups-20261010 outside the checkout. Older thesis/slide versions are preserved there; the unique local Scientific African manuscript was copied into the ignored manuscripts directory.

Stopped preliminary 30-iteration/delayed-issuance attempts are retained locally as diagnostics, excluded from the final evidence and Git synchronization. The final clock puts issuance before the target begins. No submission or journal transfer was made.

## Author-review implications

Use these new results and protocol to replace the numerical tables, abstract and discussion in the author-review manuscript. Review the target's voluntary-payoff treatment, servicing-availability assumption, residual censoring and limited temporal sample with the co-authors. The negative-result narrative must follow the corrected scores above. Existing manuscript documents have not been silently rewritten or submitted.

Reproduce with scripts/run_prospective.py, scripts/verify_prospective.py, and scripts/summarize_prospective.py. Prepare a fresh shared data directory with --terminal --prepare-only first, so both evaluations use inputs through September 2024; the runner reuses existing prepared inputs. Then run primary and --terminal evaluations against that same data directory. Protocol details and commands are in docs/prospective-rerun-20261010.md; exact input paths/hashes are in each run_manifest.json.
"""
    (r / "RERUN_REPORT.md").write_text(report)
    print(finding)
    print(score_table(pooled))
    print("Saved", r / "RERUN_REPORT.md")


if __name__ == "__main__":
    main()
