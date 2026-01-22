# Label Definition (Loan-level PD, 6-month horizon)

## Prediction task

At each month t for each active, non-defaulted loan i, predict whether the loan will enter default within the next 6 months.

## Default event

Default is defined as the first month the loan reaches 90+ days past due (DPD):
T_i = min { t : DPD_{i,t} >= 90 }.

## Binary label (6-month horizon)

For each (i,t) in the at-risk set:
y_{i,t}^{(6)} = 1 if T_i ∈ (t, t+6], else 0.

## At-risk set

Include (i,t) only if:

- loan is observable at month t,
- DPD_{i,t} < 90 (not already defaulted),
- loan is active (not closed/charged-off/sold off, if available).

## Censoring rule

If loan i is not observable for the full horizon (t, t+6], drop (i,t) from training/evaluation.

## Leakage rule

All features used at time t must be computable using information available at or before t.
No future-window aggregates overlapping (t, t+6].
