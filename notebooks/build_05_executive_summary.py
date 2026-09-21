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
            "business significance.** This page restates the headline numbers\n"
            "and the final decision; the details live in notebooks 01–04.\n"
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
            "p < 0.05 **and** lower CI bound (0.60 pp) ≥ MDE (0.50 pp).\n\n"
            "**Verdict: adopt — provisionally.** Both statistical and business\n"
            "significance are supported by the data.\n\n"
            "**Caveats that must travel with the decision:**\n"
            "1. Observational cohort → association, not proven causation.\n"
            "2. No cost data → net economic effect unchecked; gains only.\n"
            "3. `total_ads` is a suspected confounder; a randomised or adjusted\n"
            "   design is the proper follow-up.\n"
            "4. Synthetic benchmark dataset → the *method* is the reusable\n"
            "   output; absolute revenue figures are illustrative.\n"
        ),
    },
]


def main() -> None:
    out = build(CELLS, "05_executive_summary")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()