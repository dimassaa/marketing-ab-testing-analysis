"""Builds notebooks/03_hypothesis_test.ipynb (Stage 5: inferential test)."""

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
            "# 03 — Hypothesis test and confidence interval\n\n"
            "**Stage 5.** Executes the pre-registered analysis: two-proportion\n"
            "z-test (two-sided, α = 0.05), 95% confidence interval for the\n"
            "difference, assumption checks. The decision rule is read from\n"
            "`reports/pre_registration.md`, not invented after the fact.\n"
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
            "from src.inference import two_proportion_ztest, wald_ci_difference, cohens_h\n"
            "df = load_marketing_ab(root / 'data' / 'marketing_AB.csv')\n"
            "cr = conversion_rates(df).set_index('test_group')\n"
            "x_ad, n_ad = cr.loc['ad', 'converted'], cr.loc['ad', 'users']\n"
            "x_psa, n_psa = cr.loc['psa', 'converted'], cr.loc['psa', 'users']\n"
            "p_ad, p_psa = x_ad / n_ad, x_psa / n_psa\n"
            "display(cr)"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 1. Assumptions (test assumptions)\n\n"
            "Pre-registration commits us to checking these, not just running\n"
            "the test:\n"
            "- **Independence:** units are unique users (verified in Stage 1);\n"
            "  observational, no repeated measurements — plausible.\n"
            "- **Normal approximation valid:** n·p and n·(1−p) well above 5 in\n"
            "  both groups (minimum is psa converters, 420).\n"
            "- **No degenerate proportions:** both rates are ~1.8–2.6%.\n"
        ),
    },
    {
        "type": "code",
        "source": (
            "z, p_val = two_proportion_ztest(x_ad, n_ad, x_psa, n_psa)\n"
            "diff = p_ad - p_psa\n"
            "lo, hi = wald_ci_difference(p_ad, p_psa, n_ad, n_psa)\n"
            "h = cohens_h(p_ad, p_psa)\n"
            "print(f'z = {z:.2f};  two-sided p-value = {p_val:.3e}')\n"
            "print(f'diff = {diff:.5f} ({diff*100:.2f} pp);  relative uplift = {diff/p_psa:.1%}')\n"
            "print(f'95% CI for diff: [{lo:.5f}, {hi:.5f}]  = [{lo*100:.2f} pp, {hi*100:.2f} pp]')\n"
            "print(f\"Cohen's h (observed) = {h:.4f}\")"
        ),
    },
    {
        "type": "code",
        "source": (
            "fig, ax = plt.subplots(figsize=(8, 2.6))\n"
            "ax.errorbar(x=[diff], y=[0], xerr=[[diff-lo],[hi-diff]], fmt='o', ms=7, color='#4C72B0', capsize=6)\n"
            "ax.axvline(0, color='black', ls='--', lw=1)\n"
            "ax.axvline(0.005, color='#DD8452', ls=':', lw=1.5)\n"
            "ax.annotate('MDE = +0.5 pp', xy=(0.005, 0.09), ha='center', color='#DD8452')\n"
            "ax.annotate('diff = +0.77 pp', xy=(diff, 0.09), ha='center', color='#4C72B0')\n"
            "ax.set_yticks([])\n"
            "ax.set_xlabel('p_ad − p_psa (percentage points)')\n"
            "ax.set_xlim(-0.002, 0.013)\n"
            "ax.set_title('Observed difference with 95% confidence interval')\n"
            "ax.tick_params(axis='x', labelbottom=True)\n"
            "ax.xaxis.set_major_formatter(lambda x, _: f'{x*100:.1f} pp')"
        ),
    },
    {
        "type": "markdown",
        "source": (
            "## 2. Decision per the pre-registered rule\n\n"
            "| Condition (fixed in advance) | Observed | Verdict |\n"
            "|---|---|---|\n"
            "| p < 0.05 | 1.7e-13 | ✓ |\n"
            "| lower CI bound ≥ MDE (0.005) | 0.00595 | ✓ |\n\n"
            "→ **Provisional verdict: adopt.** The effect is statistically\n"
            "significant AND the 95% CI lies entirely above the business\n"
            "threshold. Stage 6 translates this into business terms (relative\n"
            "uplift, revenue impact) and states the interpretative limits.\n"
            "\n"
            "Note the honest framing: with 0.77 pp on a 1.79% base the relative\n"
            "uplift is ~43%, and the *observed* difference (0.77 pp) exceeds\n"
            "the MDE — this data does support a real, resolvable effect.\n"
        ),
    },
]


def main() -> None:
    out = build(CELLS, "03_hypothesis_test")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()