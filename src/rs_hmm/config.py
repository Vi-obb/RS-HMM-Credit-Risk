from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    figure_dir: Path
    table_dir: Path


@dataclass(frozen=True)
class SimulationConfig:
    n_months: int
    n_loans: int
    seed: int


@dataclass(frozen=True)
class RegimesConfig:
    n_regimes: int
    transition_matrix: list[list[float]]


@dataclass(frozen=True)
class InflationConfig:
    mean: list[float]
    sd: list[float]


@dataclass(frozen=True)
class PolicyRateConfig:
    base: float
    beta_inflation: float
    noise_sd: float


@dataclass(frozen=True)
class MacroConfig:
    inflation: InflationConfig
    policy_rate: PolicyRateConfig


@dataclass(frozen=True)
class LoansConfig:
    tenor_min: int
    tenor_max: int
    principal_mean: float
    principal_sd: float
    base_interest_rate: float
    beta_borrower_risk_rate: float
    beta_stress_rate: float


@dataclass(frozen=True)
class MissLogitConfig:
    alpha: float
    beta_u: float
    beta_inflation: float
    beta_policy: float
    gamma_regime: float
    delta_duration: float


@dataclass(frozen=True)
class BehaviorConfig:
    miss_logit: MissLogitConfig
    dpd_step: int
    dpd_cure_step: int
    dpd_default: int
    dpd_cap: int


@dataclass(frozen=True)
class LabelConfig:
    horizon_months: int


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float
    val_frac: float


@dataclass(frozen=True)
class EvaluationConfig:
    n_calibration_bins: int
    regime_threshold: float


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    config_path: Path
    paths: PathsConfig
    simulation: SimulationConfig
    regimes: RegimesConfig
    macro: MacroConfig
    loans: LoansConfig
    behavior: BehaviorConfig
    label: LabelConfig
    split: SplitConfig
    evaluation: EvaluationConfig


def _require(mapping: dict, key: str):
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    project_root = path.parent.parent
    raw_paths = _require(raw, "paths")
    paths = PathsConfig(
        raw_dir=_resolve_path(project_root, _require(raw_paths, "raw_dir")),
        interim_dir=_resolve_path(project_root, _require(raw_paths, "interim_dir")),
        processed_dir=_resolve_path(project_root, _require(raw_paths, "processed_dir")),
        figure_dir=_resolve_path(project_root, _require(raw_paths, "figure_dir")),
        table_dir=_resolve_path(project_root, _require(raw_paths, "table_dir")),
    )

    simulation = SimulationConfig(**_require(raw, "simulation"))
    regimes = RegimesConfig(**_require(raw, "regimes"))
    macro_raw = _require(raw, "macro")
    macro = MacroConfig(
        inflation=InflationConfig(**_require(macro_raw, "inflation")),
        policy_rate=PolicyRateConfig(**_require(macro_raw, "policy_rate")),
    )
    loans = LoansConfig(**_require(raw, "loans"))
    behavior_raw = _require(raw, "behavior")
    behavior = BehaviorConfig(
        miss_logit=MissLogitConfig(**_require(behavior_raw, "miss_logit")),
        dpd_step=int(_require(behavior_raw, "dpd_step")),
        dpd_cure_step=int(_require(behavior_raw, "dpd_cure_step")),
        dpd_default=int(_require(behavior_raw, "dpd_default")),
        dpd_cap=int(_require(behavior_raw, "dpd_cap")),
    )
    label = LabelConfig(**_require(raw, "label"))
    split = SplitConfig(**_require(raw, "split"))
    evaluation = EvaluationConfig(**_require(raw, "evaluation"))

    if regimes.n_regimes != 2:
        raise ValueError("Scaffold v1 only supports two regimes.")
    if len(regimes.transition_matrix) != regimes.n_regimes:
        raise ValueError("Transition matrix row count must match n_regimes.")
    for row in regimes.transition_matrix:
        if len(row) != regimes.n_regimes:
            raise ValueError("Transition matrix must be square.")
        if abs(sum(row) - 1.0) > 1e-6:
            raise ValueError("Each transition row must sum to 1.0.")

    return AppConfig(
        project_root=project_root,
        config_path=path,
        paths=paths,
        simulation=simulation,
        regimes=regimes,
        macro=macro,
        loans=loans,
        behavior=behavior,
        label=label,
        split=split,
        evaluation=evaluation,
    )


def ensure_output_dirs(config: AppConfig) -> None:
    for path in (
        config.paths.raw_dir,
        config.paths.interim_dir,
        config.paths.processed_dir,
        config.paths.figure_dir,
        config.paths.table_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
