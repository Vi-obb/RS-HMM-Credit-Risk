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
from sklearn.isotonic import IsotonicRegression


def time_split_by_month(df: pd.DataFrame, train_q=0.70, val_q=0.85):
    months = np.sort(df["month"].unique())
    m_train = months[int(len(months) * train_q)]
    m_val = months[int(len(months) * val_q)]
    train = df[df["month"] < m_train].copy()
    val = df[(df["month"] >= m_train) & (df["month"] < m_val)].copy()
    test = df[df["month"] >= m_val].copy()
    return train, val, test, m_train, m_val


def add_interactions(
    df: pd.DataFrame, p_col: str, base_cols: list[str], prefix: str = "x"
) -> pd.DataFrame:
    df = df.copy()
    for c in base_cols:
        df[f"{prefix}_{p_col}__{c}"] = df[p_col] * df[c]
    return df


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class SigmoidCalibrator:
    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs")

    def fit(self, p: np.ndarray, y: np.ndarray):
        x = _logit(p).reshape(-1, 1)
        self.model.fit(x, y)
        return self

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        x = _logit(p).reshape(-1, 1)
        return self.model.predict_proba(x)


class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p: np.ndarray, y: np.ndarray):
        self.model.fit(p, y)
        return self

    def predict_proba(self, p: np.ndarray) -> np.ndarray:
        p_cal = self.model.predict(p)
        p_cal = np.clip(p_cal, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p_cal, p_cal])


def make_calibrator(method: str = "sigmoid"):
    method = method.lower()
    if method == "sigmoid":
        return SigmoidCalibrator()
    if method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibration method: {method}")


def fit_and_score(
    df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    model_name: str,
    C: float = 10.0,  # loosen regularization to let interactions matter
    calibrate: bool = True,  # NEW: do val-based calibration
    calib_method: str = "sigmoid",
):
    train, val, test, m_train, m_val = time_split_by_month(df)

    X_train, y_train = train[feature_cols], train[target].astype(int)
    X_val, y_val = val[feature_cols], val[target].astype(int)
    X_test, y_test = test[feature_cols], test[target].astype(int)

    pre = ColumnTransformer([("num", StandardScaler(), feature_cols)], remainder="drop")
    base_model = LogisticRegression(max_iter=2000, solver="lbfgs", C=C)
    pipe = Pipeline([("pre", pre), ("model", base_model)])
    pipe.fit(X_train, y_train)

    # Uncalibrated probabilities
    p_train = pipe.predict_proba(X_train)[:, 1]
    p_val = pipe.predict_proba(X_val)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    def metrics(y, p, split, name):
        return {
            "model": name,
            "split": split,
            "auc": safe_auc(y, p),
            "brier": brier_score_loss(y, p),
            "logloss": log_loss(y, p),
            "event_rate": float(np.mean(y)),
            "pred_mean": float(np.mean(p)),
        }

    out_metrics = [
        metrics(y_train, p_train, "train", f"{model_name}__uncal"),
        metrics(y_val, p_val, "val", f"{model_name}__uncal"),
        metrics(y_test, p_test, "test", f"{model_name}__uncal"),
    ]

    # Metadata for exporting test preds
    meta_cols = ["loan_id", "month", "p_stress"]
    if "regime_true" in test.columns:
        meta_cols.append("regime_true")

    test_meta_uncal = test[meta_cols].copy()
    test_meta_uncal["y"] = y_test.to_numpy()
    test_meta_uncal["p"] = p_test

    test_meta_cal = None

    # Calibrated probabilities (fit calibrator on validation set)
    if calibrate:
        # Need both classes in val for calibration to make sense
        if len(np.unique(y_val)) < 2:
            print(
                f"[WARN] {model_name}: validation has single class; skipping calibration."
            )
        else:
            cal = make_calibrator(method=calib_method)
            cal.fit(p_val, y_val)

            p_train_c = cal.predict_proba(p_train)[:, 1]
            p_val_c = cal.predict_proba(p_val)[:, 1]
            p_test_c = cal.predict_proba(p_test)[:, 1]

            out_metrics.extend(
                [
                    metrics(
                        y_train, p_train_c, "train", f"{model_name}__cal_{calib_method}"
                    ),
                    metrics(y_val, p_val_c, "val", f"{model_name}__cal_{calib_method}"),
                    metrics(
                        y_test, p_test_c, "test", f"{model_name}__cal_{calib_method}"
                    ),
                ]
            )

            test_meta_cal = test[meta_cols].copy()
            test_meta_cal["y"] = y_test.to_numpy()
            test_meta_cal["p"] = p_test_c

    return out_metrics, test_meta_uncal, test_meta_cal


def main():
    labeled = pd.read_csv("data/processed/loan_monthly_panel_labeled.csv")
    loans = pd.read_csv("data/processed/loans_static.csv")
    macro_hmm = pd.read_csv("data/processed/macro_with_hmm.csv")

    df = labeled.merge(
        loans[
            ["loan_id", "principal", "tenor_months", "interest_rate", "borrower_risk"]
        ],
        on="loan_id",
        how="left",
    ).merge(macro_hmm[["month", "p_stress"]], on="month", how="left")

    df["payment_ratio"] = (df["paid_amount"] / (df["scheduled_payment"] + 1e-9)).clip(
        0.0, 1.0
    )

    target = "y_6m"

    base_common = [
        "dpd",
        "missed_payment",
        "payment_ratio",
        "principal",
        "tenor_months",
        "interest_rate",
        "borrower_risk",
    ]
    macro_cols = ["inflation", "policy_rate"]

    df = df.dropna(subset=base_common + macro_cols + ["p_stress", target]).copy()

    os.makedirs("reports/metrics", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/preds", exist_ok=True)

    results = []
    preds_to_write = {}  # filename -> df
    calibs = {}  # name -> (y, p) for overall test calibration plot

    # Helper to register prediction exports + cal curves
    def register_preds(
        name_base: str, uncal_df: pd.DataFrame, cal_df: pd.DataFrame | None
    ):
        preds_to_write[f"{name_base}__uncal_test_preds.csv"] = uncal_df
        calibs[f"{name_base}__uncal"] = (
            uncal_df["y"].to_numpy(),
            uncal_df["p"].to_numpy(),
        )
        if cal_df is not None:
            preds_to_write[f"{name_base}__cal_sigmoid_test_preds.csv"] = cal_df
            calibs[f"{name_base}__cal_sigmoid"] = (
                cal_df["y"].to_numpy(),
                cal_df["p"].to_numpy(),
            )

    # 0) baseline_nomacro
    feat_0 = base_common
    r, test_uncal, test_cal = fit_and_score(
        df, feat_0, target, "baseline_nomacro", calibrate=True, calib_method="sigmoid"
    )
    results.extend(r)
    register_preds("baseline_nomacro", test_uncal, test_cal)

    # 1) baseline_macro
    feat_1 = base_common + macro_cols
    r, test_uncal, test_cal = fit_and_score(
        df, feat_1, target, "baseline_macro", calibrate=True, calib_method="sigmoid"
    )
    results.extend(r)
    register_preds("baseline_macro", test_uncal, test_cal)

    # 2) regime_macro
    feat_2 = base_common + macro_cols + ["p_stress"]
    r, test_uncal, test_cal = fit_and_score(
        df, feat_2, target, "regime_macro", calibrate=True, calib_method="sigmoid"
    )
    results.extend(r)
    register_preds("regime_macro", test_uncal, test_cal)

    # 3) regime_macro_interactions
    df_int = add_interactions(
        df,
        "p_stress",
        base_cols=["dpd", "missed_payment", "payment_ratio"] + macro_cols,
    )
    int_cols = [c for c in df_int.columns if c.startswith("x_p_stress__")]
    feat_3 = base_common + macro_cols + ["p_stress"] + int_cols
    r, test_uncal, test_cal = fit_and_score(
        df_int,
        feat_3,
        target,
        "regime_macro_interactions",
        calibrate=True,
        calib_method="sigmoid",
    )
    results.extend(r)
    register_preds("regime_macro_interactions", test_uncal, test_cal)

    # 4) regime_nomacro_interactions
    df_int2 = add_interactions(
        df, "p_stress", base_cols=["dpd", "missed_payment", "payment_ratio"]
    )
    int_cols2 = [c for c in df_int2.columns if c.startswith("x_p_stress__")]
    feat_4 = base_common + ["p_stress"] + int_cols2
    r, test_uncal, test_cal = fit_and_score(
        df_int2,
        feat_4,
        target,
        "regime_nomacro_interactions",
        calibrate=True,
        calib_method="sigmoid",
    )
    results.extend(r)
    register_preds("regime_nomacro_interactions", test_uncal, test_cal)

    # Save metrics
    res_df = pd.DataFrame(results)
    res_df.to_csv("reports/metrics/model_comparison_v1.csv", index=False)
    print(res_df)

    # Write preds
    for fname, dfp in preds_to_write.items():
        dfp.to_csv(f"reports/preds/{fname}", index=False)

    # Calibration plot (overall test) comparing uncal vs calibrated for all models
    fig, ax = plt.subplots()
    for name, (y, p) in calibs.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Calibration (Test) — Uncalibrated vs Val-Calibrated (Sigmoid)")
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default frequency")
    ax.legend(fontsize=7)
    fig.savefig(
        "reports/figures/model_comparison_calibration_test.png",
        dpi=200,
        bbox_inches="tight",
    )

    print("Saved:")
    print(" - reports/metrics/model_comparison_v1.csv")
    print(" - reports/figures/model_comparison_calibration_test.png")
    print(" - reports/preds/*_test_preds.csv (uncal + cal_sigmoid)")


if __name__ == "__main__":
    main()
