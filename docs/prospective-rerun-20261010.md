# Corrected validation protocol, fixed before inspecting new results

This is a new evidence generation on main. Historical July/August outputs are
retained. The models, three alpha candidates, seeds, HMM covariance, and primary
Platt probability map remain the same. Differences below are methodological
corrections, so new scores must not silently replace the old experiment.

## Information clock and fitting

For servicing observation month t, issue the prediction at the start of t+2,
allowing one complete month for reporting. Labels cover t+1 through t+12 and
become available at the start of t+14. Even early-known events use that uniform
maturity date for supervised fitting. Every training, tuning and calibration
label must be available strictly before the first test issuance.

For the April 2021 observation origin: train through March 2018, tune on
April 2018–March 2019, calibrate on April 2019–March 2020, and score
April 2021–March 2022. Shift all boundaries one year for each subsequent origin.
The gap prevents immature labels entering fitting. Tuning and calibration are
disjoint, retrospective development windows; their scores are diagnostics, not
claims of prospective predictions issued at those historical dates. Only test
forecasts have the stated issuance interpretation. No training-plus-validation
refit is introduced.

HMM factors, constituent statistics, scaling and parameters fit only through
each supervised training cutoff. Filtering then proceeds forward. The final
origin is the corrected terminal-year comparison under the same information
rules. Separately score an April 2023–September 2024 terminal holdout using
that origin's unchanged training/tuning/calibration boundaries and HMM cutoff,
extending only the observation end date. This retains the original terminal test
period but removes full-window HMM fitting and immature supervised labels.
Its first 12 months must reproduce the final rolling origin's scores. The old
full-window terminal HMM results remain historical diagnostics.
They are not used to claim that only the validation schedule explains rankings.

## Macroeconomic observations

Retrieve all 12 source series from ALFRED for each forecast issuance, using the
previous day's vintage. Verify vintage suffixes in every returned series name;
cache original responses and record their URLs and SHA-256 hashes. Compute the
existing 17 indicators within that vintage and use the latest complete common
economic month no later than servicing month t. Record that source month.
Each point in the HMM sequence uses its own historical vintage, so neither
future data releases nor revisions enter earlier factor values. A common-month
vector deliberately discards fresher values from faster-release series; this
choice is fixed before evaluation. ALFRED archives measure recorded availability,
not necessarily the exact time of day of every original source release.

## Outcome and censoring

Rebuild from the calendar/sentinel-corrected unlabelled servicing panel. The
target is first observed 90+ DPD during the following 12 months before voluntary
payoff. Voluntary payoff (zero-balance code 01) resolves a negative competing
outcome. An observed default resolves a positive outcome without requiring
survival to horizon end. Require continuous records up to the resolving event
or horizon. Other exits, gaps and administrative truncation remain unresolved.
Missing delinquency status also makes the horizon unresolved unless the endpoint
is a documented voluntary payoff; unknown status in the feature lookback excludes
the prediction row. Exclude unknown labels and report monthly censoring counts. A default at
the exit month takes precedence. First default is absorbing, and prediction
rows must be before termination with complete feature lookback.

This target/population differs from the historical complete-horizon survivor
sample. It reduces that selection but does not prove censoring is ignorable.
Do not describe these complete-observed-outcome scores as validated performance
for every active loan, or as an estimate of defaults after sale/transfer.
No IPCW or causal claim is introduced. Payoff and censoring shares must accompany
the results. The sample is the existing Freddie cohorts, not a new market sample.

## Verification and uncertainty

Keep deterministic model/tuning settings, with one optimisation correction:
the inherited 30-iteration SGD cap produced tuning convergence warnings during
initial training, before test scores were assessed. Increase the cap uniformly
to 300 for the corrected experiment, retain tolerance 0.001 and seed 42, record
each tuning candidate's iterations/Brier score, and fail the run if any tuning
candidate or final classifier exhausts the cap. The stopped 30-iteration attempt
is retained locally as a diagnostic; its results are not promoted. This cap is
an algorithmic stopping check, not a proof of a global optimum. Save test-level
predictions locally with stable input-row indices for independent score checks.
Use the existing 10,000-replicate, three-month moving-block calculation, plus
predeclared 1/6/12-month block sensitivity. Intervals condition on fixed fits and
only 36 test observation months. Repeated loans/overlapping horizons remain.
Never label bootstrap sign fractions as null-hypothesis p-values.

Use the frozen uv environment, run regression and timing/censoring tests, record
source/input/lock hashes, and check pooled metrics against saved predictions.
The new results require interpretation and manuscript updates before submission.

## Worktree reconciliation

Four detached worktrees (5786, bd58, c273, ef07), based on b3084a7, contained
older versions of changes already integrated in main. Main additionally retained
the later coefficient-basis exports, calendar/sentinel corrections and explicit
input/output-path support. No older code replaced these newer implementations.
Unique historical reproducibility notes/manifests were recovered in f051f1d
and pushed to origin/main. Local Scientific African manuscript files were copied
into the ignored manuscripts directory. All four complete filesystem snapshots
and binary patches are retained outside the checkout in
../RS-HMM-Credit-Risk-worktree-backups-20261010; every archived regular file was
hash-verified before worktree removal. Differing old thesis/slide versions remain
in those backups. Only the main checkout remains registered for this repository.
