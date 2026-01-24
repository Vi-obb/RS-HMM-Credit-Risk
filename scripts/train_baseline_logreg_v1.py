from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve


def time_split_by_month(df: pd.DataFrame, train_q=0.70, val_q=0.85):
    months = np.sort(df["month"].unique())
    m_train = months[int(len(months) * train_q)]
    m_val = months[int(len(months) * val_q)]

    train = df[df["month"] < m_train].copy()
    val = df[(df["month"] >= m_train) & (df["month"] < m_val)].copy()
    test = df[df["month"] >= m_val].copy()
    return train, val, test, m_train, m_val


def main():
    # Load data
    labeled = pd.read_csv("data/processed/loan_monthly_panel_labeled.csv")
    loans = pd.read_csv("data/processed/loans_static.csv")

    # Merge static loan features
    df = labeled.merge(
        loans[
            ["loan_id", "principal", "tenor_months", "interest_rate", "borrower_risk"]
        ],
        on="loan_id",
        how="left",
    )

    # Feature engineering
    df["payment_ratio"] = df["paid_amount"] / (df["scheduled_payment"] + 1e-9)
    df["payment_ratio"] = df["payment_ratio"].clip(0.0, 1.0)

    target = "y_6m"

    feature_cols = [
        "dpd",
        "missed_payment",
        "payment_ratio",
        "inflation",
        "policy_rate",
        "principal",
        "tenor_months",
        "interest_rate",
        "borrower_risk",
    ]

    df = df.dropna(subset=feature_cols + [target]).copy()

    # Time-based split
    train, val, test, m_train, m_val = time_split_by_month(df)
    print(f"Split months at: train< {m_train}, val< {m_val}, test>= {m_val}")
    print(f"Rows: train={len(train)}, val={len(val)}, test={len(test)}")
    print(
        f"Event rate: train={train[target].mean():.4f}, val={val[target].mean():.4f}, test={test[target].mean():.4f}"
    )

    X_train, y_train = train[feature_cols], train[target].astype(int)
    X_val, y_val = val[feature_cols], val[target].astype(int)
    X_test, y_test = test[feature_cols], test[target].astype(int)

    numeric_features = feature_cols

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=2000, solver="lbfgs", C=10.0)

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    # Predict probabilities
    p_train = pipe.predict_proba(X_train)[:, 1]
    p_val = pipe.predict_proba(X_val)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    # Metrics
    def metrics(y, p, split_name):
        return {
            "split": split_name,
            "auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p),
            "logloss": log_loss(y, p),
            "event_rate": float(np.mean(y)),
            "pred_mean": float(np.mean(p)),
        }

    rows = [
        metrics(y_train, p_train, "train"),
        metrics(y_val, p_val, "val"),
        metrics(y_test, p_test, "test"),
    ]
    metrics_df = pd.DataFrame(rows)

    os.makedirs("reports/metrics", exist_ok=True)
    metrics_df.to_csv("reports/metrics/baseline_logreg_metrics.csv", index=False)
    print(metrics_df)

    # Calibration plot (test)
    frac_pos, mean_pred = calibration_curve(
        y_test, p_test, n_bins=10, strategy="quantile"
    )

    os.makedirs("reports/figures", exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Baseline Logistic Regression — Calibration (Test)")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default frequency")
    fig.savefig(
        "reports/figures/baseline_calibration_test.png", dpi=200, bbox_inches="tight"
    )

    # ROC plot (test)
    # Avoid extra deps; compute curve manually
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_test, p_test)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Baseline Logistic Regression — ROC (Test)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig.savefig("reports/figures/baseline_roc_test.png", dpi=200, bbox_inches="tight")

    # Stability: Brier score by month bucket on test
    test_eval = test[["month"]].copy()
    test_eval["y"] = y_test.to_numpy()
    test_eval["p"] = p_test

    # bucket months into ~6 bins
    test_eval["month_bin"] = pd.qcut(test_eval["month"], q=6, duplicates="drop")
    stab = (
        test_eval.groupby("month_bin")
        .apply(lambda g: brier_score_loss(g["y"], g["p"]))
        .reset_index()
    )
    stab.columns = ["month_bin", "brier"]

    stab.to_csv("reports/metrics/baseline_brier_by_timebin_test.csv", index=False)

    fig, ax = plt.subplots()
    ax.plot(range(len(stab)), stab["brier"], marker="o")
    ax.set_title("Baseline — Brier by Time Bin (Test)")
    ax.set_xlabel("Time bin (ordered)")
    ax.set_ylabel("Brier score")
    fig.savefig(
        "reports/figures/baseline_brier_timebin_test.png", dpi=200, bbox_inches="tight"
    )

    print("Saved:")
    print(" - reports/metrics/baseline_logreg_metrics.csv")
    print(" - reports/figures/baseline_calibration_test.png")
    print(" - reports/figures/baseline_roc_test.png")
    print(" - reports/metrics/baseline_brier_by_timebin_test.csv")
    print(" - reports/figures/baseline_brier_timebin_test.png")


if __name__ == "__main__":
    main()
