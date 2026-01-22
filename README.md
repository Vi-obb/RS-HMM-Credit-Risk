# Regime-Switching HMM for Loan-Level Credit Risk Under Inflation-Driven Stress

## Thesis one-liner

This thesis builds a regime-aware loan-level probability of default (PD) modeling pipeline where a Hidden Markov Model (HMM) infers latent macro-credit regimes (with special focus on inflation/rate-tightening stress), and regime information is used to improve PD calibration and stability compared to single-regime baselines.

## Why this matters

Many credit risk PD models are trained under implicit parameter stability assumptions. Under macro shocks (especially inflation and rate tightening), default dynamics can shift, degrading probability calibration and out-of-sample reliability. Regime-aware modeling aims to:

- detect latent credit conditions early (normal vs stress, or more granular regimes),
- adapt PD estimates to regime transitions,
- improve calibration and stability, not only discrimination.

## Core research questions

1. **Regime inference:** Can an HMM infer persistent latent regimes from observed macro/portfolio signals that correspond to inflation-driven stress periods?
2. **PD quality:** Does incorporating inferred regime information improve loan-level PD calibration and stability versus a single-regime PD model?
3. **Inflation-specific hypothesis:** Are PD dynamics (parameters, calibration behavior, or feature effects) materially different in an inflation/rate-tightening regime relative to a standard downturn/stress regime?

## Scope and constraints

- Primary objective is a **loan-level PD model** (borrower/loan features → PD) with regime-aware enhancement.
- Emphasis is on **probability calibration** and **stability across time/regimes**.
- Work begins **simulation-first** to validate identifiability and pipeline correctness before applying to real loan-level data.
- Real data acquisition may be uncertain; the pipeline is designed to plug in real datasets once obtained.

## Data assumptions (target real dataset)

Preferred dataset shape is a monthly panel (loan_id × month) containing:

- Borrower & loan origination features (static): amount, tenor, rate, collateral, borrower attributes, bureau/credit history.
- Behavioral features (dynamic): repayment behavior, days past due (DPD), balance/utilization, missed payments, roll rates.
- Macroeconomic series joined by time: inflation, policy rate, FX, unemployment/proxies, etc.
- Target label definition (primary): event of default (e.g., first time reaching 90+ DPD), defined carefully to avoid leakage.

## Modeling overview

### Baseline PD model (single regime)

- A transparent baseline (e.g., logistic regression scorecard-style model).
- Evaluated using discrimination + calibration:
  - AUC (supporting metric),
  - Brier score / log loss,
  - calibration curves (reliability diagrams),
  - stability checks (rolling calibration / drift).

### Regime inference (HMM switching)

- HMM fitted on observed macro/portfolio signals (and/or suitable aggregates derived from loan-level panel).
- Outputs:
  - filtered regime probabilities over time,
  - transition matrix and regime persistence,
  - interpretation of regimes (post hoc) as normal vs stress; focus on inflation-driven stress.

### Regime-aware PD model

Two main integration strategies:

1. **Soft regime feature:** include inferred regime probability as a feature in PD model.
2. **Regime-conditional PD:** allow PD parameters to differ by regime (mixture-of-experts / regime-specific models).

### Key evaluation principle

Hold everything constant (same sample splits, same borrower features) and measure the incremental value of adding regime information, especially for calibration and early-warning behavior during transitions.

## Deliverables (what “done” looks like)

- A reproducible pipeline: simulate → fit baseline → infer regimes → fit regime-aware PD → evaluate.
- Clear evidence (at least in simulation, ideally with real data) that regime-aware PD improves calibration/stability.
- A written thesis with a focused inflation-driven regime hypothesis and rigorous validation.

## Repository map (recommended)

- `configs/` experiment configs (simulation + modeling choices)
- `data/` raw/interim/processed (excluded from git except placeholders)
- `notebooks/` numbered notebooks for exploration and reproducible outputs
- `src/` reusable modules (simulation, features, models, evaluation, plotting)
- `reports/` supervisor shield, literature notes, results writeups, figures
- `experiments/` run logs, metrics tables
- `manuscript/` LaTeX thesis (template provided by department)
- `slides/` LaTeX slides (template provided by department)

## Operating rules (for humans + AI agents)

- Every session must end with at least one artifact: a plot, a table, a notebook committed, or a written paragraph.
- No “reading-only” sessions: reading must produce notes or a change in the plan.
- Keep the baseline simple and defensible before adding complexity.
- Avoid leakage: labels and features must be time-consistent.
