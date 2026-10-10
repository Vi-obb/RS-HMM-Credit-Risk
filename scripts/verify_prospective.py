"""Independent checks of scored predictions and reported results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.metrics import average_precision_score, roc_auc_score


def verify(output: Path) -> dict:
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest.get("completed") and manifest["branch"] == "main"
    metrics = pd.read_csv(output / "rolling_origin_metrics.csv")
    paired = pd.read_csv(output / "rolling_origin_paired_monthly_losses.csv")
    monthly = []
    pooled = {name: [] for name in ("baseline_logit", "macro_logit", "regime_aware_logit")}
    checks = []
    for fold in manifest["folds"]:
        prediction = pd.read_csv(output / fold["name"] / "model_predictions.csv")
        y = prediction.y_12m.to_numpy()
        assert prediction.panel_row.is_unique
        assert prediction.month_date.between(fold["test_start"], fold["test_end"]).all()
        for model in pooled:
            p = prediction[model].to_numpy()
            assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
            pooled[model].append((y, p))
            measured = {"n": len(y), "event_count": int(y.sum()), "event_rate": y.mean(), "pred_mean": p.mean(),
                        "brier": np.mean((y-p)**2), "log_loss": np.mean(-(xlogy(y,p)+xlogy(1-y,1-p))),
                        "roc_auc": roc_auc_score(y,p), "pr_auc": average_precision_score(y,p)}
            stored = metrics.loc[metrics.fold.eq(fold["name"]) & metrics.model.eq(model) & metrics.split.eq("test")].iloc[0]
            for key, value in measured.items():
                np.testing.assert_allclose(stored[key], value, rtol=1e-10, atol=1e-12)
            checks.append({"fold": fold["name"], "model": model, "prediction_metrics_match": True, "n": len(y)})
        b = prediction.baseline_logit.to_numpy()
        r = prediction.regime_aware_logit.to_numpy()
        prediction["brier_sum"] = (y-r)**2-(y-b)**2
        prediction["log_loss_sum"] = -(xlogy(y,r)+xlogy(1-y,1-r)) + (xlogy(y,b)+xlogy(1-y,1-b))
        for date, group in prediction.groupby("month_date", sort=True):
            monthly.append({"month_date": date, "n": len(group), "brier_sum": group.brier_sum.sum(), "log_loss_sum": group.log_loss_sum.sum()})
        print(f"Verified saved predictions: {fold['name']}", flush=True)
    for model, pieces in pooled.items():
        y = np.concatenate([item[0] for item in pieces]); p = np.concatenate([item[1] for item in pieces])
        stored = metrics.loc[metrics.fold.eq("pooled") & metrics.model.eq(model)].iloc[0]
        measured = {"n": len(y), "event_count": int(y.sum()), "brier": np.mean((y-p)**2),
                    "log_loss": np.mean(-(xlogy(y,p)+xlogy(1-y,1-p))),
                    "roc_auc": roc_auc_score(y,p), "pr_auc": average_precision_score(y,p)}
        for key, value in measured.items():
            np.testing.assert_allclose(stored[key], value, rtol=1e-10, atol=1e-12)
    rebuilt = pd.DataFrame(monthly).sort_values("month_date").reset_index(drop=True)
    ordered = paired.sort_values("month_date").reset_index(drop=True)
    assert rebuilt.month_date.tolist() == ordered.month_date.tolist() and len(rebuilt) == 36
    np.testing.assert_array_equal(rebuilt.n, ordered.n)
    for metric in ("brier", "log_loss"):
        np.testing.assert_allclose(rebuilt[metric+"_sum"], ordered[metric+"_loss_difference_sum"], rtol=1e-10, atol=1e-8)
    sensitivity = pd.read_csv(output / "bootstrap_block_sensitivity.csv")
    for block in (1, 3, 6, 12):
        rng = np.random.default_rng(20260727)
        for metric in ("brier", "log_loss"):
            sums = rebuilt[metric+"_sum"].to_numpy(); counts = rebuilt.n.to_numpy()
            estimates = []
            for rep in range(10_000):
                starts = rng.choice(np.arange(36-block+1), size=int(np.ceil(36/block)), replace=True)
                indices = (starts[:,None] + np.arange(block)[None,:]).ravel()[:36]
                estimates.append(sums[indices].sum()/counts[indices].sum())
            row = sensitivity.loc[sensitivity.block_months.eq(block) & sensitivity.metric.eq(metric)].iloc[0]
            np.testing.assert_allclose([row.ci_lower_95, row.ci_upper_95], np.quantile(estimates,[.025,.975]), rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(row.estimate_regime_minus_baseline, sums.sum()/counts.sum(), rtol=1e-10, atol=1e-12)
    info = pd.read_csv(output / "information_audit.csv")
    assert len(info) == 9 and info.all_labels_mature.all()
    assert (pd.to_datetime(info.latest_label_available) < pd.to_datetime(info.first_test_issue)).all()
    coverage = pd.read_csv(output / "outcome_coverage.csv")
    for fold in manifest["folds"]:
        n = coverage.loc[coverage.month_date.between(fold["test_start"],fold["test_end"]),"known_outcome"].sum()
        reported = metrics.loc[metrics.fold.eq(fold["name"]) & metrics.split.eq("test"),"n"]
        assert reported.eq(n).all()
    seeds = pd.read_csv(output / "rolling_origin_hmm_initialisations.csv")
    paths = pd.read_csv(output / "rolling_origin_hmm_paths.csv")
    assert len(seeds) == 15
    for fold, group in seeds.groupby("fold"):
        candidates = group.loc[group.converged & np.isfinite(group.train_log_likelihood)]
        expected = candidates.sort_values(["train_log_likelihood","seed"],ascending=[False,True]).iloc[0].seed
        assert paths.loc[paths.fold.eq(fold),"selected_seed"].eq(expected).all()
    result = {"completed": True, "checks": checks, "pooled_metrics_from_saved_predictions": True,
              "paired_losses_from_saved_predictions": True, "bootstrap_sensitivity_independently_reproduced": True,
              "maturity_and_population_counts_verified": True, "hmm_seed_selection_verified": True}
    (output / "verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    verify(parser.parse_args().output_dir)
