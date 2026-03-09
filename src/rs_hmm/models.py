from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rs_hmm.splits import time_split_by_month


BASE_FEATURES = [
    "dpd",
    "missed_payment",
    "payment_ratio",
    "principal",
    "tenor_months",
    "interest_rate",
    "borrower_risk",
]
MACRO_FEATURES = ["inflation", "policy_rate"]
REGIME_SOFT = ["p_stress"]
REGIME_HARD = ["hard_stress"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: list[str]
    interaction_base: list[str] | None = None


MODEL_SPECS = [
    ModelSpec(name="baseline", features=BASE_FEATURES),
    ModelSpec(name="macro", features=BASE_FEATURES + MACRO_FEATURES),
    ModelSpec(name="regime_hard", features=BASE_FEATURES + MACRO_FEATURES + REGIME_HARD),
    ModelSpec(name="regime_soft", features=BASE_FEATURES + MACRO_FEATURES + REGIME_SOFT),
    ModelSpec(
        name="regime_interactions",
        features=BASE_FEATURES + MACRO_FEATURES + REGIME_SOFT,
        interaction_base=["dpd", "missed_payment", "payment_ratio", "inflation", "policy_rate"],
    ),
]


def _augment_interactions(df: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    interaction_cols = []
    for column in base_cols:
        new_col = f"p_stress_x_{column}"
        out[new_col] = out["p_stress"] * out[column]
        interaction_cols.append(new_col)
    return out, interaction_cols


def prepare_model_frame(
    labeled: pd.DataFrame,
    loans: pd.DataFrame,
    macro_hmm: pd.DataFrame,
    regime_threshold: float,
    horizon: int,
) -> pd.DataFrame:
    df = labeled.merge(
        loans[["loan_id", "principal", "tenor_months", "interest_rate", "borrower_risk"]],
        on="loan_id",
        how="left",
    ).merge(
        macro_hmm[["month", "p_stress", "hard_stress"]],
        on="month",
        how="left",
    )

    df["hard_stress"] = (df["p_stress"] >= regime_threshold).astype(int)
    df["payment_ratio"] = (df["paid_amount"] / (df["scheduled_payment"] + 1e-9)).clip(0.0, 1.0)
    target = f"y_{horizon}m"
    keep_cols = BASE_FEATURES + MACRO_FEATURES + ["p_stress", "hard_stress", target]
    return df.dropna(subset=keep_cols).copy()


def train_all_models(
    df: pd.DataFrame,
    target: str,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df, test_df = time_split_by_month(df, train_frac, val_frac)
    predictions: list[pd.DataFrame] = []
    coefficients: list[dict] = []

    split_map = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for spec in MODEL_SPECS:
        model_df = df.copy()
        features = list(spec.features)
        if spec.interaction_base:
            model_df, interaction_cols = _augment_interactions(model_df, spec.interaction_base)
            features.extend(interaction_cols)
            train_df, val_df, test_df = time_split_by_month(model_df, train_frac, val_frac)
            split_map = {"train": train_df, "val": val_df, "test": test_df}

        preprocessor = ColumnTransformer(
            [("num", StandardScaler(), features)],
            remainder="drop",
        )
        classifier = LogisticRegression(max_iter=2000, solver="lbfgs")
        pipeline = Pipeline([("pre", preprocessor), ("model", classifier)])
        pipeline.fit(split_map["train"][features], split_map["train"][target].astype(int))

        coef_values = pipeline.named_steps["model"].coef_[0]
        for feature, coef in zip(features, coef_values):
            coefficients.append({"model": spec.name, "feature": feature, "coefficient": float(coef)})

        for split_name, split_df in split_map.items():
            preds = pipeline.predict_proba(split_df[features])[:, 1]
            predictions.append(
                pd.DataFrame(
                    {
                        "model": spec.name,
                        "split": split_name,
                        "loan_id": split_df["loan_id"].to_numpy(),
                        "month": split_df["month"].to_numpy(),
                        "y": split_df[target].astype(int).to_numpy(),
                        "p": preds,
                        "p_stress": split_df["p_stress"].to_numpy(),
                        "hard_stress": split_df["hard_stress"].to_numpy(),
                        "regime_true": split_df.get("regime_true", pd.Series(index=split_df.index, dtype=float)).to_numpy(),
                    }
                )
            )

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(coefficients)
