# Pre-registration: ad vs psa conversion experiment

**Date:** 2026-09-21
**Status:** registered BEFORE running the inferential test (Stage 5).
**Analyst:** (project learning portfolio)
**Dataset:** Marketing A/B Testing, `data/marketing_AB.csv` (588,101 unique users).

This document fixes the analysis plan in advance. Nothing below is informed
by the observed outcome of the hypothesis test (none has been run yet).

## 1. Metric and unit of analysis

- **Metric:** conversion rate (share of users with `converted = 1`).
- **Unit of analysis:** user id (verified: 588,101 rows = 588,101 unique users).

## 2. Hypotheses

| | |
|---|---|
| Null hypothesis (H₀) | p_ad = p_psa  (no difference in conversion rate) |
| Alternative hypothesis (H₁) | p_ad ≠ p_psa  (two-sided) |

Two-sided because the campaign could plausibly move conversion in either
direction; we do not pre-commit to an ad-only uplift.

## 3. Test design

- **Method:** two-proportion z-test (normal approximation of the binomial).
- **Significance level:** α = 0.05 (two-sided), so the rejection region is split
  across both tails. Type I error budget = 5%.
- **Statistical power (target):** 80%, i.e. β = 0.20. Under-powered tests are
  not interpretable as "no effect" — they only mean "not detected".
- **Minimum detectable effect (MDE):** +0.5 percentage points (baseline ~1.8%
  control conversion ⇒ ≈ +28% relative). MDE is the smallest business-relevant
  uplift we commit to detecting. Chosen over the raw +2 p.p. from the original
  plan because +2 p.p. (≈+110% relative) is implausible in this context and
  makes the power analysis trivial.
- **Baseline conversion for planning:** p_psa = 1.79% (observed control rate,
  used only for the sample-size plan; it is the pre-experiment baseline).

## 4. Sample-size plan

- Compute per-group n for the two-sided z-test at (α = 0.05, power = 0.80,
  MDE = 0.005) using the binomial sample-size formula.
- The data are heavily imbalanced (ad 96% / psa 4%). The binding constraint is
  the **smaller** group (psa, 23,524 users): if the formula's per-group n ≤
  min(n_ad, n_psa), the experiment is adequately powered for the MDE.
- Report also a power curve (n → power) and sensitivity of required n to MDE.

## 5. Decision rule (fixed in advance)

| Condition | Verdict |
|---|---|
| z-test p < 0.05 **and** effect ≥ MDE (lower CI bound ≥ 0) | statistically AND practically significant → adopt |
| p < 0.05 but effect < MDE (unimportant in business terms) | statistically significant, practically negligible → **do not adopt** |
| p ≥ 0.05 | H₀ not rejected → do not adopt (and report achieved power / CI) |

Statistical significance alone never triggers adoption; practical significance
(MDE and confidence interval in business terms) is required.

## 6. Reporting

- Report two-sided 95% confidence interval for p_ad − p_psa, plus relative
  uplift and business impact estimates.
- Check test assumptions (independence, n·p ≥ 5, no degenerate proportions)
  and record them in the report.
- Interpretation is explicitly limited to association (observational cohort);
  causality is not claimed. `total_ads`, `most_ads_day/hour` are flagged as
  potential confounders to discuss, not to adjust for post hoc.

## 7. Reproducibility

- env: Python 3.12.3 in `.venv`, versions pinned in `requirements.txt`.
- All calculations execute from `src/` functions covered by pytest.
- Random state is not needed (exact methods only; no bootstrap/permutation).