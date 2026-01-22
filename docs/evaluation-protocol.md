# Evaluation Protocol (Calibration-first)

## Splitting

Use time-based splits (chronological) to avoid leakage and to test generalization across regimes:

- Train: earliest portion of time
- Validation: middle portion
- Test: latest portion of time

## Primary metrics (calibration)

- Brier score
- Calibration curves (reliability diagrams)

## Secondary metrics (discrimination)

- AUC-ROC

## Stability analysis

Report calibration and Brier score:

- by time buckets (e.g., yearly/quarterly)
- by inferred regime (once regime inference is implemented)
Focus: performance degradation during regime transitions.
