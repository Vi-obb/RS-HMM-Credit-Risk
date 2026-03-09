# A Regime-Aware Model for Loan-Level PD

The thesis will not be framed as a generic "use HMMs for consumer credit risk" project. That angle is now too broad to be novel on its own. The sharper contribution is an interpretable, calibration-focused regime-aware PD framework that studies how borrower risk sensitivities and probability estimates change across macro regimes, especially during inflation and rate-tightening stress.

## Core Thesis Direction

The planned thesis asks whether inferred macro regimes can improve loan-level PD estimation in ways that matter for practice:

- better probability calibration,
- more stable performance across time and regime transitions,
- interpretable regime-dependent borrower sensitivities,
- and defensible validation before real-data deployment.

## Proposed Novelty

The thesis should lean on the following contribution claims:

- A calibration-first view of regime-aware PD modeling rather than a pure prediction-accuracy story.
- An interpretable logistic-style PD framework, not just a black-box regime-aware classifier.
- Explicit comparison of multiple regime-integration strategies:
  - macro variables only,
  - hard regime labels,
  - soft regime probabilities,
  - regime-feature interactions,
  - and, if feasible, regime-conditional models.
- A simulation-first identifiability study showing when latent-regime inference helps and when it fails.
- A focused inflation/rate-tightening stress hypothesis instead of an unrestricted macro story.
- Analysis of transition dynamics and stress persistence, not only average performance by regime.

## Research Questions

1. Does adding inferred macro-regime information improve loan-level PD calibration relative to borrower-only and macro-augmented baselines?
2. Are any gains concentrated in stress periods or around regime transitions?
3. Do soft regime probabilities outperform hard regime assignments for calibration and stability?
4. Do borrower risk sensitivities vary materially across inferred regimes?
5. Under what data-generating conditions can the latent regime be recovered reliably enough to support downstream PD modeling?

## Methodological Position

- Primary modeling family: transparent logistic PD models.
- Regime inference: two-state HMM on macro series, with scope to extend only if justified.
- Primary evaluation metrics: Brier score and calibration curves.
- Secondary metrics: AUC-ROC, log loss, and transition-period diagnostics.
- Validation principle: chronological splits only, with explicit leakage control.
- Development principle: simulation before real-data adaptation.

## Workspace Layout

- `src/rs_hmm/`: reusable package code
- `scripts/`: reproducible command-line entrypoints
- `configs/`: experiment settings
- `notebooks/`: notebook-first exploration built on package functions
- `data/`: raw, interim, and processed artifacts
- `reports/`: saved figures and metrics tables
- `manuscript/`: thesis LaTeX scaffold
- `slides/`: presentation LaTeX scaffold
- `notes/`: literature and research logs

## Commands

Run the synthetic workflow from the repository root:

```bash
PYTHONPATH=src python -m scripts.simulate --config configs/sim_v1.yml
PYTHONPATH=src python -m scripts.build_labels --config configs/sim_v1.yml
PYTHONPATH=src python -m scripts.fit_hmm --config configs/sim_v1.yml
PYTHONPATH=src python -m scripts.train_models --config configs/sim_v1.yml
PYTHONPATH=src python -m scripts.evaluate_models --config configs/sim_v1.yml
```

The same workflow is exposed through the notebooks, but the scripts remain the reproducible source of truth.

## Operating Rules

- Keep the baseline simple before adding model complexity.
- Treat calibration and robustness as first-class outcomes.
- Avoid unverifiable novelty claims; make the contribution narrow and defensible.
- Every new component should answer a specific research question.
- Do not add real-data assumptions to the pipeline without documenting leakage and censoring implications.
