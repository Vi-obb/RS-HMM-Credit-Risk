# A Regime-Aware Model for Loan-Level Probability of Default

This repository documents an empirical mortgage credit risk study built on Freddie Mac sample loan data and monthly FRED macroeconomic data. The thesis examines whether a latent macro stress signal improves loan-level probability of default estimation for residential mortgages while remaining interpretable and useful in practice.

## Active Thesis Direction

The thesis studies whether a latent macro stress signal can improve loan-level probability of default estimation for residential mortgages in a way that remains interpretable and useful for practice.

The data sources are:

- Freddie Mac sample origination and servicing files for cohorts `2015` through `2025`
- FRED monthly inflation, interest-rate, and unemployment series for the same broad period

The modeling flow is:

1. Build a monthly loan-level mortgage panel from Freddie Mac origination and servicing files.
2. Use monthly FRED inflation, Fed Funds, and unemployment data as inputs to a two-state HMM.
3. Estimate the filtered probability of being in a macro stress regime at each month.
4. Carry that stress probability into a loan-level logistic PD model through borrower-feature interactions.
5. Compare the regime-aware model against two simpler logit benchmarks.

## Model Set

The active study will compare exactly three models:

- `baseline_logit`: borrower and loan features only
- `macro_logit`: borrower and loan features plus raw macro variables
- `regime_aware_logit`: borrower and loan features plus filtered `p_stress` and `p_stress x borrower feature` interactions

This keeps the empirical comparison focused. The thesis is not being positioned as a replication of Brookfield's gradient boosting approach. Brookfield remains related work, not an implementation target.

## Label And Evaluation

The primary label is a 12-month transition to `90+` days past due among loans that are not already `90+` DPD at month `t`.

Primary evaluation emphasis:

- Brier score
- calibration plots
- calibration by regime and time slice

Secondary metrics:

- ROC-AUC
- PR-AUC
- log loss

## Current Repository Status

The repository centers on the Freddie Mac + FRED mortgage PD study.

- The current Python pipeline under `src/rs_hmm/` and `scripts/` remains a legacy synthetic scaffold.
- The previous synthetic notebooks, generated figures, generated tables, and synthetic interim or processed data are archived under `legacy/simulation_v1/`.
- Active `reports/` directories remain placeholders until the empirical workflow is populated.

The repository documents the empirical workflow and keeps the archived synthetic path available for reference only.

## Implementation Roadmap

### Phase 1: Documentation Alignment

- Align the README, manuscript, slides, and notes with the mortgage PD thesis narrative.
- Keep the active thesis story centered on the Freddie Mac and FRED workflow.
- Describe the empirical workflow clearly in the project documentation.

### Phase 2: Legacy Archive

- Preserve the synthetic thesis path under `legacy/simulation_v1/`.
- Keep legacy material accessible for background only, not as active thesis evidence.

### Phase 3: Empirical Build

1. Parse Freddie Mac sample origination and servicing files for `2015` through `2025`.
2. Construct the mortgage loan-month panel and the 12-month serious-delinquency label.
3. Fit the macro HMM on inflation, Fed Funds, and unemployment.
4. Train the three approved logistic PD models using chronological splits.
5. Produce calibration-first evaluation tables and figures.

### Phase 4: Code Migration

- Replace simulation-first notebook names, workflow assumptions, and config structure with empirical equivalents.
- Promote the empirical pipeline to active status after it is implemented and validated.

## Repo Layout

- `data/`: Freddie Mac and macro source data, plus placeholder interim and processed directories
- `legacy/simulation_v1/`: archived synthetic notebooks, outputs, and document snapshots
- `manuscript/`: thesis chapter scaffolding for the empirical narrative
- `notes/`: research notes, presentation script, and implementation roadmap
- `reports/`: placeholder location for future empirical outputs
- `scripts/` and `src/rs_hmm/`: legacy synthetic scaffolding
- `slides/`: active presentation deck for the mortgage PD thesis

## Immediate Next Steps

- finalize the empirical variable list and borrower feature set
- define terminal servicing states and censoring rules for the 12-month label
- replace the synthetic data-prep path with Freddie Mac ingestion
- rebuild notebooks and reports around the empirical workflow
