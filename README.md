# Marketing A/B Experiment — Statistical Analysis

A complete, reproducible A/B testing cycle on a 588K-user marketing dataset:
statistical power planning, pre-registration, hypothesis testing, and
business-significance evaluation.

[Русская версия](README_ru.md)

![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![Tests](https://img.shields.io/badge/tests-25%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![MDE](https://img.shields.io/badge/pre--registration-yes-blue)

![Observed difference with 95% confidence interval](assets/result_ci.png)

> **TL;DR — verdict: do not adopt.** The pooled ad advantage (+0.77 p.p.,
> p = 1.7e-13) is statistically significant **but not uniform**: a `total_ads`
> stratification check (notebook 06) shows the effect is concentrated in
> high-exposure users and plausibly reflects a confounder. Recommendation:
> run a randomised test before any real rollout. Full reasoning:
> `reports/05_executive_summary.html`.

## Table of Contents

- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Installation and Usage](#detailed-installation-and-usage)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Pre-registration](#pre-registration)
- [Methodology](#methodology)
- [Results](#results)
- [Business Impact](#business-impact)
- [Testing](#testing)
- [Limitations](#limitations)
- [Recommendations](#recommendations)
- [Support](#support)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This project simulates a day in the life of a product analyst: it evaluates
whether an advertising campaign (`ad` group) changes purchase conversion
compared with a public-service announcement baseline (`psa` group). The
analysis follows data-analysis golden standards: data integrity is verified
before any conclusion, the test plan is fixed before the results are seen
(pre-registration), and decisions require both statistical and practical
significance.

Main contributions:

- Pre-registered decision rule (MDE, alpha, power), so no part of the analysis
  is tuned to a found result.
- Closed-form sample size, power curve, and MDE sensitivity analysis, all
  implemented as tested `src/` functions.
- Two-sided z-test for proportions with Wald confidence interval, checked
  against `statsmodels`.
- Business impact quantification with honest limits (observational cohort,
  no cost data, an *evidenced* confounder via the `total_ads` stratification
  check in notebook 06).

In a "day in the life", every analyst task maps to a tested artifact:

| Task | Where it lives |
| ---- | -------------- |
| Data integrity check (unit of analysis, duplicates, missing, ranges) | `src/eda.py`, notebook 01 |
| Power and sample-size planning *before* the data is judged | `src/inference.py`, notebook 02 |
| Pre-registration of hypotheses and MDE | `reports/pre_registration.md` |
| Confirmatory hypothesis test and confidence intervals | notebook 03 |
| Business-significance translation | notebook 04 |
| Confounder robustness (stratified check) | notebook 06 |
| Stakeholder communication (executive summary) | `reports/05_executive_summary.html`, this README |

The full reasoning behind every stage (why this approach, key terms,
pitfalls) is documented in `notes/` (Russian, English key terms included).

## Tech Stack

| Category            | Technologies                                            |
| ------------------- | ------------------------------------------------------- |
| Language            | Python 3.12                                             |
| Statistics          | SciPy, statsmodels (cross-validation of formulas)       |
| Data processing     | pandas, NumPy                                           |
| Visualization       | Matplotlib, seaborn                                     |
| Notebooks           | Jupyter (nbformat + nbclient reproducible builds)       |
| Testing             | pytest                                                  |
| Environment         | uv / venv, `requirements.txt` with pinned versions      |

## Project Structure

```
├── src/                  Reusable, tested analysis functions (eda, inference)
├── tests/                pytest coverage for src/
├── notebooks/            Stages 01–06: EDA, planning, test, business, summary, robustness (built from build_*.py, outputs embedded)
├── reports/              Pre-registration plan + executive summary (HTML)
├── assets/               PNG figures reused by the README
├── notes/                Russian learning notes per stage
├── data/                 Dataset (source documented; CSV not committed, 22 MB)
├── README.md             This file
└── requirements.txt      Pinned Python dependencies
```

## Quick Start

**Option A: uv (recommended)**

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest tests/
```

**Option B: classic venv + pip**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

**Option C: explore the reports without installing anything**

Open `reports/05_executive_summary.html` in a browser. It contains the full
decision plus reproducible headline numbers.

## Detailed Installation and Usage

Prerequisites: Python 3.12, `uv` (or `pip`), and the dataset placed at
`data/marketing_AB.csv` (see [Data](#data)).

Rebuild and re-execute every notebook from the tested `src/` functions:

```bash
for f in notebooks/build_0*.py; do .venv/bin/python "$f"; done
```

Export the executive summary to a standalone HTML report:

```bash
.venv/bin/python -m jupyter nbconvert --to html \
  --output-dir reports notebooks/05_executive_summary.ipynb
```

Open a notebook interactively:

```bash
.venv/bin/python -m jupyter notebook notebooks/05_executive_summary.ipynb
```

## Data

Marketing A/B Testing dataset (Kaggle, faviovaz). Source and redownload
instructions are in `data/README.md`.

| Stat                          | Value            |
| ----------------------------- | ---------------- |
| Rows                          | 588,101          |
| Unique users                  | 588,101 (no duplicates) |
| Missing values                | 0                |
| Groups                        | ad = 564,577 (96%), psa = 23,524 (4%) |
| Conversion (ad / psa)         | 2.55% / 1.79%    |
| `total_ads` range             | 1–2065 per user  |
| `most_ads_day` / `most_ads_hour` | all 7 days, hours 0–23 |

Features: `user_id`, `test_group`, `converted` (0/1), `total_ads`,
`most_ads_day`, `most_ads_hour`.

Cleaning: nothing to drop — a stray Kaggle index column is removed and
`converted` is typed as integer 0/1 (see `src/eda.py`).

## Exploratory Data Analysis

- Unit of analysis is the user: every row is exactly one user id.
- Groups are heavily imbalanced (96/4); the psa sample (23,524) is the binding
  constraint for power planning.
- Observed conversion differs descriptively between groups (see the chart) —
  no inference is drawn from EDA alone.

![Conversion rate by test group](assets/eda_conversion_by_group.png)

> [!NOTE]
> The single 0.77 p.p. EDA difference is a descriptive fact about the sample,
> not a significance claim. Significance is decided by the pre-registered
> test, which is run only after the plan is fixed.

## Pre-registration

The decision rule was written into `reports/pre_registration.md` at planning
time, before any outcome was analysed; nothing in this project is tuned to
the results.

> **Registered plan (excerpt):** H₀: p_ad = p_psa; H₁: p_ad ≠ p_psa
> (two-sided). α = 0.05, target power = 0.80, MDE = +0.5 p.p., baseline
> p_psa = 1.79%. Decision rule: adopt only if p < 0.05 **and** the effect
> reaches the MDE — statistical significance alone never triggers adoption.

The experiment was adequately powered *before* the analysis ran:

| Planning result | Value |
| --------------- | ----- |
| Required n per group at 80% power | 12,547 |
| psa group size (binding constraint) | 23,524 |
| Achieved power at the psa-group size | 0.97 |

![Power vs group size](assets/power_curve.png)

## Methodology

Closed-form per-group sample size for the pre-registered two-sided
two-proportion z-test (parameters as fixed above):

$$n = \left( \frac{z_{1-\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} +
z_{1-\beta}\sqrt{p_0(1-p_0)+p_1(1-p_1)}}{p_1-p_0} \right)^2$$

Where p1 = p0 + MDE, p_bar = (p0 + p1)/2.

Sensitivity analysis: resolving a +0.1 p.p. effect would need ~283K per group
(infeasible with the available psa sample); +1 p.p. needs only ~3.5K. The
observed effect is well inside the resolvable range.

![Required n vs MDE](assets/sensitivity_mde.png)

The inference itself is a two-proportion z-test with a pooled variance under
H0, plus a 95% Wald confidence interval for the difference. Assumptions
(independence, n*p and n*(1-p) well above 5 in both groups) are checked and
reported before the result.

## Results

| Item | Value |
| ---- | ----- |
| Conversion, ad vs psa | 2.55% vs 1.79% |
| Difference (p.p.) | +0.77 |
| Relative uplift | +43.1% |
| z statistic / two-sided p-value | 7.37 / 1.7e-13 |
| 95% CI for the difference | [+0.60, +0.94] p.p. |
| Additional converters across all users (95% CI) | [3,500, 5,548] |

Decision, **before** the robustness check, applied strictly from the
pre-registered rule: p < 0.05 **and** the lower CI bound (+0.60 p.p.) at
least the MDE (+0.50 p.p.) — both hold for the pooled difference.

**But the pooled effect is not uniform (notebook 06).** Stratifying by
`total_ads` shows the ad advantage is concentrated in high-exposure users
(21+ ads, especially 50+, where 63% of ad conversions live); among users with
1–20 ads the difference is not significant. Removing the 50+ stratum drops
the pooled difference from +0.77 to about +0.15 p.p. (not significant).

![Per-stratum ad advantage with 95% CIs](assets/strata_diff_ci.png)

Note the honest reading of the chart: in the low/mid strata the CIs are wide
or the cells are too sparse (expected events < 5) for a test — so this is
evidence of a *concentrated* effect, not strong evidence that ad is worse
there.

**Verdict: do not adopt as a uniform effect.** A single causal ad advantage
is not supported by this observational data — the signal is plausibly the
confounder `total_ads` (exposure is not randomised). A rollout requires a
randomised experiment or a properly adjusted (causal) analysis first.

## Business Impact

- Relative uplift of +43.1% on a 1.79% base, i.e. roughly 4,524 additional
  converters over the full 588,101-user population (95% CI 3.5K–5.5K).
- The dataset has no revenue column, so money impact is a declared what-if
  (revenue per converter, RPC):

| RPC (currency units) | Low (95% CI) | Point estimate | High (95% CI) |
| -------------------- | ------------ | -------------- | ------------- |
| 5                    | 17,499       | 22,620         | 27,741        |
| 10                   | 34,997       | 45,239         | 55,481        |
| 20                   | 69,995       | 90,479         | 110,963       |

These are illustrative scenarios (the dataset is synthetic), not a real
company's P&L.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

18 tests cover both modules: loading and integrity checks
(`tests/test_eda.py`) and the statistical functions (`tests/test_inference.py`),
including cross-validation of the z-test and the sample-size formula against
`statsmodels`.

## Limitations

- Observational cohort: the campaign already ran; the test shows association,
  not proven causation.
- No ad-cost data: only the upside is quantified; a final decision needs the
  cost side (marginal economics).
- `total_ads` is an **evidenced** confounder, not just a suspicion: the pooled
  +0.77 p.p. does not survive stratification (notebook 06). A randomised or
  properly adjusted (causal) design is required, not optional.
- The benchmark data is synthetic — the reusable output is the method and the
  pipeline, not the absolute business figures.

## Recommendations

- Run a prospective randomised test (or an adjusted observational analysis on
  `total_ads`) before any real rollout — the robustness evidence in notebook
  06 is the reason this is mandatory, not precautionary.
- Add revenue and ad-cost tracking to the experiment logging; re-evaluate
  with real economics.
- Consider a sequential testing (group sequential design) framework if the
  campaign budget constrains the sample size.
- Report the achieved power (0.97) alongside any future insignificant
  results to keep interpretations honest.

## Support

- Found a bug or have a question: open a GitHub issue.
- General discussion about the methodology: GitHub Discussions.

## Contributing

Contributions are welcome. Please open an issue first to discuss the intended
change, keep the `src/` functions tested, and regenerate the notebooks with
the `build_*.py` scripts before submitting a pull request.

## License

[MIT](LICENSE)

## Acknowledgements

- Dataset: Marketing A/B Testing by faviovaz,
  [Kaggle](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing).
- Cross-validation of statistical formulas: `statsmodels`.
- Charts and layout follow the analysis-quality rules from the project's
  learning notes (minimal, question-driven visualization).