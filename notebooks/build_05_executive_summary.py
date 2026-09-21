"""Builds notebooks/05_executive_summary.ipynb (Stage 7: conclusion page)."""

from __future__ import annotations

import nbformat

from builders import build

_ANCHOR = (
    "import sys, pathlib\n"
    "root = pathlib.Path.cwd()\n"
    "while root != root.parent and not (root / 'data' / 'marketing_AB.csv').exists():\n"
    "    root = root.parent\n"
    "if str(root) not in sys.path: sys.path.insert(0, str(root))\n"
)

CELLS: list[dict[str, str]] = [
    {
        "type": "markdown",
        "source": (
            "# 05 — Executive summary\n\n"
            "**Full A/B cycle: power planning → pre-registration → inference →\n"
            "business significance → robustness.** This page restates the\n"
            "headline numbers and the final decision; the details live in\n"
            "notebooks 01–04 and the robustness check in notebook 06.\n\n"
            "**Status: SUMMARY / DECISION.** Reviews the confirmatory and\n"
            "robustness evidence together and states the final verdict.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            _ANCHOR
            + "from src.eda import load_marketing_ab, conversion_rates\n"
            "from src.inference import two_proportion_ztest, wald_ci_difference, sample_size_proportions\n"
            "df = load_marketing_ab(root / 'data' / 'marketing_AB.csv')\n"
            "cr = conversion_rates(df).set_index('test_group')\n"
            "x_ad, n_ad = cr.loc['ad', 'converted'], cr.loc['ad', 'users']\n"
            "x_psa, n_psa = cr.loc['psa', 'converted'], cr.loc['psa', 'users']\n"
            "p_ad, p_psa = x_ad / n_ad, x_psa / n_psa\n"
            "z, pval = two_proportion_ztest(x_ad, n_ad, x_psa, n_psa)\n"
            "lo, hi = wald_ci_difference(p_ad, p_psa, n_ad, n_psa)\n"
            "n_req = sample_size_proportions(0.0179, 0.005)\n"
            "import pandas as pd\n"
            "rows = [\n"
            "    ('Required n per group (MDE +0.5 pp, 80% power)', f'{n_req:,}'),\n"
            "    ('psa group size (binding constraint)', '23,524'),\n"
            "    ('Observed conversion rate ad vs psa', '2.55% vs 1.79%'),\n"
            "    ('Difference (pp)', '+0.77'),\n"
            "    ('Relative uplift', '+43.1%'),\n"
            "    ('z statistic / two-sided p-value', f'{z:.2f} / {pval:.1e}'),\n"
            "    ('95% CI for difference (pp)', '[+0.60, +0.94]'),\n"
            "    ('Additional converters (95% CI)', '[3,500, 5,548]'),\n"
            "]\n"
            "display(pd.DataFrame(rows, columns=['Item', 'Value']).style.hide(axis='index'))"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## Final decision\n\n"
            "Per the pre-registered rule (reports/pre_registration.md):\n"
            "p < 0.05 **and** lower CI bound (0.60 pp) ≥ MDE (0.50 pp) — the\n"
            "pooled difference is statistically significant on its own.\n\n"
            "**But the pooled result is not uniform (see notebook 06):** the ad\n"
            "advantage concentrates in high-exposure users (21+ ads, especially\n"
            "50+, where 63% of ad conversions live). Among users with 1–20 ads\n"
            "the difference is not significant. Removing the 50+ stratum drops\n"
            "the pooled difference from +0.77 to about +0.15 pp (not\n"
            "significant).\n\n"
            "**Verdict: do not adopt as a uniform effect.** A single, causal\n"
            "ad advantage is **not supported** by this observational data: the\n"
            "signal is plausibly the confounder `total_ads` (exposure is not\n"
            "randomised). A rollout requires a randomised experiment or a\n"
            "properly adjusted (causal) analysis first.\n\n"
            "**Caveats that must travel with the decision:**\n"
            "1. Observational cohort → association, not proven causation.\n"
            "2. No cost data → net economic effect unchecked; gains only.\n"
            "3. `total_ads` moves from *suspected* to *evidenced* confounder\n"
            "   (notebook 06): the naive pooled effect does not survive\n"
            "   stratification.\n"
            "4. Synthetic benchmark dataset → the *method* is the reusable\n"
            "   output; absolute revenue figures are illustrative.\n"
            "5. Low-exposure strata are sparse (expected events < 5) — the\n"
            "   absence of a detected effect there is read descriptively, not\n"
            "   as proof of zero effect.\n\n"
            "Walk through the full analysis: notebooks 01–04, robustness in\n"
            "notebook 06, and this summary.\n"
        ),
    },
]


def main() -> None:
    out = build(CELLS, "05_executive_summary")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()