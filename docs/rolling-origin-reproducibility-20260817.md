# Rolling-Origin Reproducibility Record

## Scope and evidence status

The repository now tracks the recovered rolling-origin implementation and its
command-line runner. The implementation incorporates both later data audits:

- Freddie Mac unavailable-value sentinels are converted to missing values with
  additive missingness indicators; medians are fitted on training data only,
  and regime interactions are formed after imputation.
- Only internal calendar gaps are inserted; observations are excluded when the
  feature lookback or 12-month outcome horizon crosses a gap; first `90+` DPD is
  absorbing; and zero-balance effective month is used with reporting-month
  fallback for termination.

The active comparison remains exactly `baseline_logit`, `macro_logit`, and
`regime_aware_logit`. Platt calibration is fitted on each fold's validation
window. This workflow does not inspect a test fold to add models, features,
thresholds, covariance types, calibration methods, or tuning candidates.

The tracked July result directories remain historical evidence and are not
overwritten. The calendar-plus-sentinel corrected candidate and verification
directories are a distinct evidence generation. They become authoritative only
when every gate below passes for the exact tracked implementation and locked
environment.

## Exact invocation

Run twice into two new, empty directories. Do not point either run at
`reports/tables/` or at a prior result directory.

```bash
uv sync --frozen

uv run python scripts/run_rolling_freeze.py \
  --panel /absolute/path/to/loan_monthly_panel_labeled.csv \
  --macro /absolute/path/to/macro_monthly.csv \
  --output-dir /absolute/path/to/candidate

uv run python scripts/run_rolling_freeze.py \
  --panel /absolute/path/to/loan_monthly_panel_labeled.csv \
  --macro /absolute/path/to/macro_monthly.csv \
  --output-dir /absolute/path/to/verification
```

The audited combined run used these input identities:

| Input | SHA-256 |
| --- | --- |
| Corrected labeled panel | `7afe5444411659582cb753b2b387f442974585cb9dadbfb083a76cc38d12dfae` |
| Transformed monthly macro panel | `cec0943e7fd613ea6c2a885948418ed2c14705bd548c8179b1868f32dd34aa5f` |
| `uv.lock` | `b58224941676c5fb1c7468871e065d5fa5b2d4001d467a48b4477512d7765236` |
| `pyproject.toml` | `44b2da36da50b39a88423741b005892148ea5c794fb200fbd2cbefa04cbadc48` |

The recovered rolling implementation has SHA-256
`fcbec9060dd092cbc2a86e478440aad08de5da33c272d9464043fdaae1ff35ab`;
the runner has SHA-256
`33e33a83d9338415c302b51eb54465090fbfddbb40edb5ddf5defb8909ce4158`.
The complete audited source/test hash set and the eleven expected output hashes
are in `reports/manifests/calendar_sentinel_fix_20260817.sha256`.

## Frozen protocol

- Folds: train through `2020-03`, `2021-03`, and `2022-03`; validation is the
  following 12 months; test is the subsequent non-overlapping 12 months,
  ending `2022-03`, `2023-03`, and `2024-03` respectively.
- HMM: two states, full covariance, at most 500 iterations, deterministic starts
  `11`, `23`, `42`, `71`, and `101`. Factor statistics, scaling, and HMM fitting
  use only data through the fold's training cutoff.
- HMM-start rejection: reject a start that raises `ValueError` or
  `FloatingPointError`, does not converge, or produces a non-finite training
  log likelihood. Among valid starts, select the highest training log
  likelihood, breaking a tie in favor of the lower seed. Record every start in
  `rolling_origin_hmm_initialisations.csv`; do not silently drop a rejected
  start or replace it with a newly chosen seed.
- Models: only the three frozen logistic specifications. Tune SGD regularisation
  over `3e-6`, `1e-5`, and `3e-5` using validation Brier score, with deterministic
  sampling and classifier seed `42`.
- Missing values: fit median imputers on training rows only, retain the additive
  missingness indicators, then construct `p_stress` interactions from imputed
  values. Fit feature scaling on training rows only.
- Calibration: fit Platt scaling on raw validation predictions and apply that
  fixed map to validation and test predictions. It is the primary and only
  probability map in this freeze workflow.
- Uncertainty: paired regime-aware-minus-baseline losses over the 36 test months;
  10,000 moving-block resamples; three-month blocks; sampling with replacement;
  deterministic seed `20260727`; observation-count-weighted loss differences;
  percentile 95% interval.

## Authoritative-rerun acceptance checklist

- [ ] Record the base Git commit and confirm the worktree contains no unrelated
  source changes.
- [ ] Verify the panel, macro, lockfile, project file, runner, and rolling module
  SHA-256 values before execution.
- [ ] Record OS, architecture, Python, `uv`, NumPy, pandas, SciPy,
  scikit-learn, hmmlearn, statsmodels, and PyYAML versions. Use
  `uv sync --frozen`; a different environment is a new portability check, not a
  byte-reproduction claim.
- [ ] Confirm the three fold boundary rows exactly match the frozen protocol.
- [ ] Confirm all five HMM starts per fold are present, their errors/convergence
  state is explicit, and the selected seed follows the training-likelihood rule.
- [ ] Confirm each model's selected regularisation, training-only imputation
  medians, missing shares, and Platt intercept/slope are recorded in model
  selection output.
- [ ] Confirm the paired uncertainty table records 36 months, block length 3,
  10,000 replicates, and seed `20260727` for both losses.
- [ ] Require all eleven CSVs in each run and reject unexpected extra CSVs.
- [ ] First require byte identity between independent candidate and verification
  runs using `cmp`. Then compare their SHA-256 values to the tracked manifest.
- [ ] If byte identity fails, compare schemas, row order, keys, integer columns,
  and floating columns with explicit tolerances. A numerical-only match must be
  labeled as such and explained; it is not the frozen authoritative result.
- [ ] Inspect fold-level and pooled tables for completeness and finite values.
  This is verification only: test results must not trigger new model search.
- [ ] Promote corrected outputs only by an explicit research decision. Preserve
  the historical July files and document which evidence generation is cited.

## Verification commands

```bash
find /absolute/path/to/candidate -maxdepth 1 -name '*.csv' -print | sort
find /absolute/path/to/verification -maxdepth 1 -name '*.csv' -print | sort

for file in /absolute/path/to/candidate/*.csv; do
  name="$(basename "$file")"
  cmp "$file" "/absolute/path/to/verification/$name"
done

shasum -a 256 /absolute/path/to/candidate/*.csv
uv run --with pytest pytest -q
git diff --check
```

The combined audit's candidate and verification outputs were independently
generated and are byte-identical for all eleven CSVs. That result is preserved
as evidence; this note does not promote or rewrite current report tables.
