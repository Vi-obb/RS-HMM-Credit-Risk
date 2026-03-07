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

- Primary modeling family: transparent logistic / scorecard-style PD models.
- Regime inference: two-state HMM on macro series, with scope to extend only if justified.
- Primary evaluation metrics: Brier score and calibration curves.
- Secondary metrics: AUC-ROC, log loss, and transition-period diagnostics.
- Validation principle: chronological splits only, with explicit leakage control.
- Development principle: simulation before real-data adaptation.

## Immediate Next Build Order

1. Create a new config file and rebuild a minimal simulation pipeline from config.
2. Reintroduce label construction with strict time-consistency rules.
3. Add simple borrower-only and macro-augmented logistic baselines.
4. Add HMM-based regime inference and posterior extraction.
5. Compare hard-state, soft-probability, and interaction-based regime-aware PD variants.
6. Add calibration-by-time, calibration-by-regime, and transition-window evaluation.
7. Document identifiability and failure modes before moving to real data.

## Operating Rules

- Keep the baseline simple before adding model complexity.
- Treat calibration and robustness as first-class outcomes.
- Avoid unverifiable novelty claims; make the contribution narrow and defensible.
- Every new component should answer a specific research question.
- Do not add real-data assumptions to the pipeline without documenting leakage and censoring implications.
