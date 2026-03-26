# A Regime-Aware Mortgage PD Thesis Reset

This repository is being reset around an empirical mortgage credit risk study built on Freddie Mac sample loan data and monthly FRED macroeconomic data. The active thesis direction is no longer a simulation-first study. The current goal is to document a clear research narrative, archive the obsolete synthetic path, and prepare the repository for an empirical implementation phase.

## Active Thesis Direction

The thesis studies whether a latent macro stress signal can improve loan-level probability of default estimation for residential mortgages in a way that remains interpretable and useful for practice.

The planned data sources are:

- Freddie Mac sample origination and servicing files for cohorts `2015` through `2025`
- FRED monthly inflation, interest-rate, and unemployment series for the same broad period

The intended modeling flow is:

1. Build a monthly loan-level mortgage panel from Freddie Mac origination and servicing files.
2. Use monthly FRED inflation, Fed Funds, and unemployment data as inputs to a two-state HMM.
3. Estimate the filtered probability of being in a macro stress regime at each month.
4. Carry that stress probability into a loan-level logistic PD model through borrower-feature interactions.
5. Compare the regime-aware model against two simpler logit benchmarks.

## Planned Model Set

The active study will compare exactly three models:

- `baseline_logit`: borrower and loan features only
- `macro_logit`: borrower and loan features plus raw macro variables
- `regime_aware_logit`: borrower and loan features plus filtered `p_stress` and `p_stress x borrower feature` interactions

This keeps the empirical comparison focused. The thesis is not being positioned as a replication of Brookfield's gradient boosting approach. Brookfield remains related work, not an implementation target.

## Planned Label And Evaluation

The primary label is planned as a 12-month transition to `90+` days past due among loans that are not already `90+` DPD at month `t`.

Primary evaluation emphasis:

- Brier score
- calibration plots
- calibration by regime and time slice

Secondary metrics:

- ROC-AUC
- PR-AUC
- log loss

## Current Repository Status

The repository is in a narrative-reset phase.

- Active documents now describe the Freddie Mac + FRED thesis direction.
- The current Python pipeline under `src/rs_hmm/` and `scripts/` remains a legacy synthetic scaffold.
- The previous synthetic notebooks, generated figures, generated tables, and synthetic interim or processed data have been archived under `legacy/simulation_v1/`.
- Active `reports/` directories have been returned to placeholder status until the empirical workflow is built.

This means the repository currently documents the intended empirical workflow, but it does not yet implement that workflow end to end.

## Implementation Roadmap

### Phase 1: Documentation Reset

- Align the README, manuscript, slides, and notes with the empirical mortgage PD narrative.
- Remove simulation-first claims from the active thesis story.
- Document the empirical workflow in future-state language where code has not yet been migrated.

### Phase 2: Legacy Archive

- Preserve the old synthetic thesis path under `legacy/simulation_v1/`.
- Keep legacy material accessible for background only, not as active thesis evidence.

### Phase 3: Empirical Build

1. Parse Freddie Mac sample origination and servicing files for `2015` through `2025`.
2. Construct the mortgage loan-month panel and the 12-month serious-delinquency label.
3. Fit the macro HMM on inflation, Fed Funds, and unemployment.
4. Train the three approved logistic PD models using chronological splits.
5. Produce calibration-first evaluation tables and figures.

### Phase 4: Code Migration

- Replace simulation-first notebook names, workflow assumptions, and config structure with empirical equivalents.
- Promote the empirical pipeline to active status only after it exists and has been validated.

## Repo Layout

- `data/`: Freddie Mac and macro source data, plus placeholder interim and processed directories
- `legacy/simulation_v1/`: archived synthetic notebooks, outputs, and document snapshots
- `manuscript/`: thesis chapter scaffolding for the empirical narrative
- `notes/`: research notes, presentation script, and implementation roadmap
- `reports/`: placeholder location for future empirical outputs
- `scripts/` and `src/rs_hmm/`: legacy synthetic scaffolding pending migration
- `slides/`: active presentation deck for the reset thesis direction

## Immediate Next Steps

- finalize the empirical variable list and borrower feature set
- define terminal servicing states and censoring rules for the 12-month label
- replace the synthetic data-prep path with Freddie Mac ingestion
- rebuild notebooks and reports around the empirical workflow
