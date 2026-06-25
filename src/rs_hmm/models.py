from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rs_hmm.empirical import FREDDIE_NUMERIC_FEATURES, align_macro_months
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
EMPIRICAL_MACRO_FEATURES = ["inflation", "policy_rate", "unemployment"]
REGIME_SOFT = ["p_stress"]
REGIME_HARD = ["hard_stress"]
EMPIRICAL_INTERACTION_BASE = [
    "credit_score",
    "original_cltv",
    "original_dti",
    "original_ltv",
    "original_interest_rate",
]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    features: list[str]
    interaction_base: list[str] | None = None


MODEL_SPECS = [
    ModelSpec(name="baseline", features=BASE_FEATURES),
    ModelSpec(name="macro", features=BASE_FEATURES + MACRO_FEATURES),
    ModelSpec(
        name="regime_aware",
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
    loans: pd.DataFrame | None,
    macro_hmm: pd.DataFrame,
    regime_threshold: float,
    horizon: int,
) -> pd.DataFrame:
    df = labeled.copy()
    if "month_date" in df.columns:
        df["month_date"] = pd.to_datetime(df["month_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if loans is not None and {"loan_id", "principal", "tenor_months", "interest_rate", "borrower_risk"}.issubset(loans.columns):
        df = df.merge(
            loans[["loan_id", "principal", "tenor_months", "interest_rate", "borrower_risk"]],
            on="loan_id",
            how="left",
        )

    macro_hmm = align_macro_months(macro_hmm, df)
    macro_cols = ["month", "p_stress", "hard_stress"] + [
        column for column in EMPIRICAL_MACRO_FEATURES if column in macro_hmm.columns
    ]
    if "month_date" in macro_hmm.columns:
        macro_cols.append("month_date")
    macro_cols = list(dict.fromkeys(macro_cols))

    merge_keys = ["month_date"] if "month_date" in df.columns and "month_date" in macro_hmm.columns else ["month"]
    df = df.merge(macro_hmm[macro_cols], on=merge_keys, how="left", suffixes=("", "_macro"))
    if "month_macro" in df.columns and "month" not in merge_keys:
        df = df.drop(columns=["month_macro"])

    df["hard_stress"] = (df["p_stress"] >= regime_threshold).astype(int)
    if {"paid_amount", "scheduled_payment"}.issubset(df.columns):
        df["payment_ratio"] = (df["paid_amount"] / (df["scheduled_payment"] + 1e-9)).clip(0.0, 1.0)
    target = f"y_{horizon}m"
    keep_cols = _required_model_columns(df) + ["p_stress", "hard_stress", target]
    return df.dropna(subset=keep_cols).copy()


def _is_empirical_frame(df: pd.DataFrame) -> bool:
    return {"credit_score", "original_upb", "original_interest_rate"}.issubset(df.columns)


def _available(columns: pd.Index, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in columns]


def _empirical_model_specs(df: pd.DataFrame) -> list[ModelSpec]:
    base = _available(df.columns, FREDDIE_NUMERIC_FEATURES)
    macro = _available(df.columns, EMPIRICAL_MACRO_FEATURES)
    interactions = _available(df.columns, EMPIRICAL_INTERACTION_BASE)
    return [
        ModelSpec(name="baseline_logit", features=base),
        ModelSpec(name="macro_logit", features=base + macro),
        ModelSpec(
            name="regime_aware_logit",
            features=base + macro + REGIME_SOFT,
            interaction_base=interactions,
        ),
    ]


def _model_specs_for_frame(df: pd.DataFrame) -> list[ModelSpec]:
    if _is_empirical_frame(df):
        return _empirical_model_specs(df)
    return MODEL_SPECS


def _required_model_columns(df: pd.DataFrame) -> list[str]:
    required: list[str] = []
    for spec in _model_specs_for_frame(df):
        required.extend(spec.features)
        if spec.interaction_base:
            required.extend(spec.interaction_base)
    return list(dict.fromkeys(required))


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

    empirical = _is_empirical_frame(df)
    for spec in _model_specs_for_frame(df):
        model_df = df
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
        classifier = (
            SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30, tol=1e-3, random_state=42)
            if empirical
            else LogisticRegression(max_iter=2000, solver="lbfgs")
        )
        pipeline = Pipeline([("pre", preprocessor), ("model", classifier)])
        pipeline.fit(split_map["train"][features], split_map["train"][target].astype(int))

        coef_values = pipeline.named_steps["model"].coef_[0]
        for feature, coef in zip(features, coef_values):
            coefficients.append({"model": spec.name, "feature": feature, "coefficient": float(coef)})

        for split_name, split_df in split_map.items():
            preds = pipeline.predict_proba(split_df[features])[:, 1]
            prediction_frame = pd.DataFrame(
                {
                    "model": spec.name,
                    "split": split_name,
                    "loan_id": split_df["loan_id"].to_numpy(),
                    "month": split_df["month"].to_numpy(),
                    "y": split_df[target].astype(int).to_numpy(),
                    "p": preds,
                    "p_stress": split_df["p_stress"].to_numpy(),
                    "hard_stress": split_df["hard_stress"].to_numpy(),
                }
            )
            if "month_date" in split_df.columns:
                prediction_frame["month_date"] = split_df["month_date"].to_numpy()
            if "cohort_year" in split_df.columns:
                prediction_frame["cohort_year"] = split_df["cohort_year"].to_numpy()
            if "regime_true" in split_df.columns:
                prediction_frame["regime_true"] = split_df["regime_true"].to_numpy()
            predictions.append(prediction_frame)

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(coefficients)
