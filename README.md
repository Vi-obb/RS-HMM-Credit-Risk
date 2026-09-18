# A Regime-Aware Model for Loan-Level Probability of Default

This repository contains the replication code, data processing pipelines, and empirical evaluation framework for the paper:

> **A Regime-Aware Model for Loan-Level Probability of Default**  
> *Vincent Fiifi Obbeng, M. A. Boateng, and Y. E. Ayekple*  
> Department of Mathematics, Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana

---

## Overview

Credit risk models frequently struggle during macroeconomic transitions, where aggregate stress shifts default probabilities nonlinearly across borrower risk segments. This project presents an interpretable, regime-aware framework for estimating loan-level Probability of Default (PD) on residential mortgages:

1. **Macro Regime Identification**: A two-state Hidden Markov Model (HMM) estimates a filtered latent macro stress probability ($p_{\text{stress}}$) from monthly macroeconomic indicators.
2. **Loan-Level Integration**: The filtered stress signal is carried into borrower-level logistic default models through structured interactions with credit profile and leverage metrics (e.g., credit score, debt-to-income ratio, loan-to-value ratio).
3. **Empirical Benchmarking**: The regime-aware specification is compared against a baseline borrower logit and an explicit macro-augmented logit using rolling chronological evaluation splits, with a primary focus on probability calibration (Brier score, calibration curves) and discrimination (ROC-AUC, PR-AUC).

---

## Data Sources

The empirical analysis is based on two publicly accessible data sources:

1. **Freddie Mac Single-Family Loan-Level Dataset**
   - Publicly accessible to registered users via the [Freddie Mac Research Portal](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).
   - Provides loan-level origination characteristics and monthly performance histories for fixed-rate residential mortgages.

2. **Federal Reserve Economic Data (FRED)**
   - Openly available from the [Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/).
   - Provides monthly macroeconomic time series, including the Consumer Price Index (CPI / Inflation), Effective Federal Funds Rate (EFFR), and Civilian Unemployment Rate (UNRATE).
