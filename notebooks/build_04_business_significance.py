"""Builds notebooks/04_business_significance.ipynb (Stage 6)."""

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
            "# 04 — Practical significance and business impact\n\n"
            "**Stage 6.** Translates the statistically significant result into\n"
            "business terms: relative uplift, additional converters across the\n"
            "user base, and an illustrative revenue scenario.\n\n"
            "**Status: INTERPRETATION (post-hoc, decision-support).** Context\n"
            "for the fixed confirmatory verdict in notebook 03, not a second\n"
            "test: the hypothesis and its threshold are not re-chosen here.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            _ANCHOR
            + "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "sns.set_theme(style='whitegrid')\n"
            "from src.eda import load_marketing_ab, conversion_rates\n"
            "from src.inference import wald_ci_difference\n"
            "df = load_marketing_ab(root / 'data' / 'marketing_AB.csv')\n"
            "cr = conversion_rates(df).set_index('test_group')\n"
            "x_ad, n_ad = cr.loc['ad', 'converted'], cr.loc['ad', 'users']\n"
            "x_psa, n_psa = cr.loc['psa', 'converted'], cr.loc['psa', 'users']\n"
            "p_ad, p_psa = x_ad / n_ad, x_psa / n_psa\n"
            "diff = p_ad - p_psa\n"
            "lo, hi = wald_ci_difference(p_ad, p_psa, n_ad, n_psa)\n"
            "users = len(df)\n"
            "print(f'relative uplift: {diff/p_psa:.1%}')\n"
            "print(f'additional converters (all {users:,} users): {diff*users:,.0f}')\n"
            "print(f'CI: [{lo*users:,.0f}, {hi*users:,.0f}]')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 1. Business-wide impact with uncertainty\n\n"
            "One number without a CI hides the risk. The interval answers:\n"
            "*if the ad runs for the whole population, how many additional\n"
            "converters should we expect?* (3.5K–5.5K).\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "fig, ax = plt.subplots(figsize=(8, 2.8))\n"
            "ax.errorbar(x=[diff*users], y=[0], xerr=[[diff*users - lo*users],[hi*users - diff*users]],\n"
            "            fmt='o', ms=8, color='#4C72B0', capsize=6, capthick=2)\n"
            "ax.annotate(f'+{diff*users:,.0f} converters', xy=(diff*users, 0.12), ha='center', color='#4C72B0')\n"
            "ax.set_yticks([])\n"
            "ax.set_xlabel('additional converters (95% CI) across all users')\n"
            "ax.set_title('Business impact of showing the ad (vs psa)')\n"
            "ax.set_xlim(2000, 7000)"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 2. Illustrative revenue scenario\n\n"
            "The dataset contains **no revenue/money column**, so revenue cannot\n"
            "be read from the data: it is a deliberately declared scenario\n"
            "(assumption: revenue per converter, RPC). Every row is a 'what if'.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "rpc_values = [5, 10, 20]\n"
            "rows = []\n"
            "for rpc in rpc_values:\n"
            "    rows.append({'revenue_per_converter': rpc,\n"
            "                 'low (95% CI)': rpc * lo * users,\n"
            "                 'point estimate': rpc * diff * users,\n"
            "                 'high (95% CI)': rpc * hi * users})\n"
            "rev = pd.DataFrame(rows).set_index('revenue_per_converter')\n"
            "rev.index.name = 'RPC, currency units'\n"
            "display(rev.round(0).astype(int))"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 3. Honest interpretation limits (statistics ≠ causation)\n\n"
            "- **Observational cohort:** the campaign already ran; the comparison\n"
            "  is association, not proof that showing the ad *causes* conversion.\n"
            "- **Cost side is missing:** gains only; no ad spend, no marginal\n"
            "  economics. A decision needs cost data.\n"
            "- **Confounder warning:** `total_ads` correlates with both exposure\n"
            "  and conversion; a rigorous causal design (randomisation or\n"
            "  adjustment) is the follow-up this dataset cannot provide.\n"
            "- **Synthetic data:** numbers are illustrative of the *method*; the\n"
            "  absolute business figures are not a real company's P&L.\n"
            "\n"
            "Resume takeaway: the decision pipeline (MDE → power → CI → business\n"
            "scenario) is fully reproducible, the limits are owned explicitly.\n"
        ),
    },
]


def main() -> None:
    out = build(CELLS, "04_business_significance")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()